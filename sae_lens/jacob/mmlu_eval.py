import sys
sys.path.append("/root/sae-data-selection/SAELens/")
from datasets import load_dataset, Dataset
from transformer_lens import utils, HookedTransformer
import gc
import torch
from tqdm import tqdm
import pandas as pd
from typing import Optional
import plotly.express as px

from less.jacob.load_sae_from_hf import load_sae_from_hf
from sae_lens.config import DTYPE_MAP

torch.set_grad_enabled(False)



# Data loading functions
NUM_TO_STR = {0: "A", 1: "B", 2: "C", 3: "D"}
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

def str_to_num(answer_str):
    if answer_str == "A" or answer_str == "a" or answer_str == " A" or answer_str == " a":
        return 0
    elif answer_str == "B" or answer_str == "b" or answer_str == " B" or answer_str == " b":
        return 1
    elif answer_str == "C"  or answer_str == "c" or answer_str == " C" or answer_str == " c":
        return 2
    elif answer_str == "D" or answer_str == "d" or answer_str == " D" or answer_str == " d":
        return 3
    else: # model did not predict any of the choices
        return 'invalid_answer'


# Evaluation functions
def get_mmlu_accuracy(model : HookedTransformer, 
                      sae_list : Optional[list] = None,
                      subject : str = "high_school_physics",
                      ):
    '''
    model (HookedTransformer): the model to evaluate
    sae_list (list): list of SAEs whose sum we patch into the model. If None, the model is evaluated without patching.
    subject (str): name of the subject-specific MMLU dataset, e.g. "abstract_algebra"
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

    # Run model with SAE patching to get accuracy
    num_correct = 0
    num_invalid = 0
    for (prompt, answer) in tqdm(zip(formatted_prompts, correct_answers)):
        output_token_id = model.run_with_hooks(
            prompt, 
            fwd_hooks=[(hook_pt, patch_hook)]
            )[0, -1].argmax()
        output_answer_id = str_to_num(model.tokenizer.decode(output_token_id))
        if output_answer_id == answer:
            num_correct += 1
        elif output_answer_id == 'invalid_answer':
            num_invalid += 1
    print(f"model output invalid answer for {num_invalid/len(formatted_prompts)/100:.1f}% of prompts")
    return num_correct / len(formatted_prompts)
