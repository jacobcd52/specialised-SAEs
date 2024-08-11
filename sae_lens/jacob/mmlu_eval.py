import sys
sys.path.append("/root/sae-data-selection/SAELens/")
from datasets import load_dataset, Dataset
from transformer_lens import utils, HookedTransformer
import gc
import torch
from tqdm import tqdm
import pandas as pd
from typing import Optional, Tuple
import plotly.express as px

from sae_lens.jacob.load_sae_from_hf import load_sae_from_hf
from sae_lens.config import DTYPE_MAP


torch.set_grad_enabled(False)



# Data loading functions
NUM_TO_STR = {0: "A", 1: "B", 2: "C", 3: "D"}
STR_TO_NUM = {"A": 0, "B": 1, "C": 2, "D": 3}

def format_prompt(question, choices):
    formatted_choices = "\n".join([f"{NUM_TO_STR[i]}. {choice}" for i, choice in enumerate(choices)])
    prompt = f"<start_of_turn>user\nQuestion:\n{question}\n\nChoices:\n{formatted_choices}\n<end_of_turn>\n"
    return prompt

def format_answer(answer_index):
    return f"<start_of_turn>model\nThe correct answer is: {NUM_TO_STR[answer_index]}<end_of_turn>\n"

def create_multi_shot_prompt(data: Dataset, num_shots=5):
    multi_shot_prompt = ""
    for i in range(num_shots):
        question = data['question'][i]
        choices = data['choices'][i]
        answer = data['answer'][i]
        
        multi_shot_prompt += format_prompt(question, choices)
        multi_shot_prompt += format_answer(answer)
        
        if i < num_shots - 1:
            multi_shot_prompt += "\n"  # Add a newline between examples, except for the last one
    
    return multi_shot_prompt

def prepare_dataset(data: Dataset, num_shots=5):
    multi_shot_prompt = create_multi_shot_prompt(data, num_shots=num_shots)
    
    formatted_prompts = []
    correct_answers = []
    
    for i in range(num_shots, len(data['question'])):  # Leave out the items we used for the multi-shot prompt
        question = data['question'][i]
        choices = data['choices'][i]
        answer = data['answer'][i]
        
        formatted_prompt = format_prompt(question, choices)
        full_prompt = multi_shot_prompt + "\n" + formatted_prompt + "\n" + "<start_of_turn>model\nThe correct answer is:"
        
        formatted_prompts.append(full_prompt)
        correct_answers.append(answer)
    
    return formatted_prompts, correct_answers

# Evaluation functions
def get_mmlu_accuracy(model : HookedTransformer, 
                      sae_list : Optional[list] = None,
                      subject : str = "high_school_physics",
                      ) -> Tuple[float, float]:
    '''
    model (HookedTransformer): the model to evaluate
    sae_list (list): list of SAEs whose sum we patch into the model. If None, the model is evaluated without patching.
    subject (str): name of the subject-specific MMLU dataset, e.g. "abstract_algebra"
    
    Returns:
    Tuple[float, float]: (accuracy, average_correct_probability)
    '''

    # Get hook point
    if sae_list:
        # Make sure all SAEs were trained at same hook point
        all_hook_pts = set([sae.cfg.hook_name for sae in sae_list])
        assert len(all_hook_pts) == 1, "All models must have the same hook point"
        hook_pt = all_hook_pts.pop()
    else:
        # Set a dummy hook point (this will not be used)
        hook_pt = "blocks.0.hook_resid_pre"

    # Create hook function to patch sum of SAE reconstructions into model
    def patch_hook(act, hook):
        if sae_list:
            recons = torch.zeros_like(act)
            for sae in sae_list:
                recons += sae(act) # may need to change if SAE output is formatted weirdly
            return recons
        else:
            return act
    
    # Load and reformat the dataset
    data = load_dataset("cais/mmlu", subject, split="test") 
    formatted_prompts, correct_answers = prepare_dataset(data)

    # Define all possible variations of answer options
    answer_variations = {
        "A": ["A", "a", " A", " a"],
        "B": ["B", "b", " B", " b"],
        "C": ["C", "c", " C", " c"],
        "D": ["D", "d", " D", " d"]
    }

    # Run model with SAE patching to get accuracy and average correct probability
    num_correct = 0
    total_correct_prob = 0.0
    
    for (prompt, answer) in tqdm(zip(formatted_prompts, correct_answers)):
        logits = model.run_with_hooks(
            prompt, 
            fwd_hooks=[(hook_pt, patch_hook)]
            )[0, -1]
        
        # Get probabilities for all variations of A, B, C, D
        option_probs = torch.zeros(4)
        for i, option in enumerate(["A", "B", "C", "D"]):
            option_prob = 0
            for variation in answer_variations[option]:
                variation_token_id = model.tokenizer.encode(variation)[-1]
                option_prob += logits[variation_token_id].exp().item()
            option_probs[i] = option_prob
        
        # Normalize probabilities
        option_probs = option_probs / option_probs.sum()
        
        # Get the predicted answer and its probability
        predicted_answer = NUM_TO_STR[option_probs.argmax().item()]
        correct_prob = option_probs[answer].item()
        
        # Update accuracy and total correct probability
        if STR_TO_NUM[predicted_answer] == answer:
            num_correct += 1
        total_correct_prob += correct_prob
    
    accuracy = num_correct / len(formatted_prompts)
    avg_correct_prob = total_correct_prob / len(formatted_prompts)
    
    return accuracy, avg_correct_prob
