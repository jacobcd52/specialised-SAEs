import sys
sys.path.append("/root/specialised-SAEs/")
import os
import logging
from datasets import load_dataset, Dataset
import torch
import json
import gc
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils
from huggingface_hub import login, HfApi, create_repo

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.sae import SAE

logger = logging.getLogger()
logger.setLevel(logging.ERROR)
login(token="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")
api = HfApi()

torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")


def filter_and_reupload(subject : str,
                        n_ctx : int = 1024,
                        batch_size : int = 32,
                        threshold : float = 4.0,
                        ):
    
    # Load the data
    data = load_dataset(f"jacobcd52/{subject}")
    tokenized_data = utils.tokenize_and_concatenate(data["train"], model.tokenizer, max_length=n_ctx)
    tokens = tokenized_data["tokens"].cuda()
    del tokenized_data
    gc.collect()
    torch.cuda.empty_cache()
    n_batches = tokens.shape[0] // batch_size

    # Calculate perplexity and filter data
    filtered_data = []

    for b in tqdm(range(n_batches)):
        batch = tokens[b*batch_size:(b+1)*batch_size].cuda()
        loss = model(batch, return_type="loss", loss_per_token=True).mean(dim=1)
        
        for i in range(batch_size):
            if loss[i] <= threshold:
                filtered_data.append({
                    "text": model.tokenizer.decode(batch[i].tolist(), skip_special_tokens=True),
                })

            else:
                print(f"Batch {b} element {i} Loss {loss[i].item():.2f}. Filtered out: ", model.tokenizer.decode(batch[i].tolist(), skip_special_tokens=True))
    # Create a new dataset with the filtered data
    cleaned_dataset = Dataset.from_dict({
        "text": [item["text"] for item in filtered_data],
    })

    # Set up Hugging Face repository
    repo_id = f"jacobcd52/{subject}_cleaned"

    # Check if the repository exists
    api = HfApi()
    try:
        api.repo_info(repo_id, repo_type="dataset")
        print(f"Repository {repo_id} already exists.")
    except Exception:
        print(f"Repository {repo_id} does not exist. Creating new repository...")
        create_repo(repo_id, repo_type="dataset", private=False)

    # Save the cleaned dataset locally as a JSON file
    local_file = "cleaned_dataset.json"
    with open(local_file, "w") as f:
        json.dump(cleaned_dataset.to_dict(), f)

    # Upload the file to the Hugging Face repository
    try:
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo="cleaned_dataset.json",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Dataset successfully uploaded to {repo_id}")
    except Exception as e:
        print(f"An error occurred while uploading the dataset: {str(e)}")
        print("Please check your permissions and connection, then try again.")

    # Clean up the local file
    os.remove(local_file)


subject_thresholds = {
    "hs_bio" : 4.0,
    "hs_math" : 4.2,
    "hs_phys" : 4.2,
    "college_bio" : 4.1,
    "college_math" : 4.3,
    "college_phys" : 4.3,
    "history" : 4.0,
    "econ" : 4.0
}

for subject, threshold in subject_thresholds.items():
    filter_and_reupload(subject, threshold=threshold)
    
