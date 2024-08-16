import os
from huggingface_hub import hf_hub_download
from sae_lens.sae import SAE
import torch 
from sae_lens.config import DTYPE_MAP
from safetensors import safe_open
from safetensors.torch import save_file

def load_sae_from_hf(repo_id, filename, cfg_filename, device="cuda", dtype="float32"):
    # Make a directory to store the weights and cfg
    temp_gsae_path = "temp_sae"
    os.makedirs(temp_gsae_path, exist_ok=True)

    # Define the local paths for the files
    temp_weights_path = os.path.join(temp_gsae_path, "sae_weights.safetensors")
    temp_cfg_path = os.path.join(temp_gsae_path, "cfg.json")

    try:
        # Download weights
        downloaded_weights_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_dir=temp_gsae_path
        )

        # Filter out weights containing "gsae" in their keys
        with safe_open(downloaded_weights_path, framework="pt", device="cpu") as f:
            original_state_dict = {k: f.get_tensor(k) for k in f.keys()}

        filtered_state_dict = {k: v for k, v in original_state_dict.items() if "gsae" not in k}

        # Save the filtered weights
        save_file(filtered_state_dict, temp_weights_path)

        # Download cfg
        downloaded_cfg_path = hf_hub_download(
            repo_id=repo_id, 
            filename=cfg_filename, 
            local_dir=temp_gsae_path
        )
        os.rename(downloaded_cfg_path, temp_cfg_path)
    except Exception as e:
        print(f"Error downloading weights or cfg: {e}")
        return None

    # Load weights into GSAE
    sae = SAE.load_from_pretrained(temp_gsae_path, device=device, dtype=dtype)
    return sae