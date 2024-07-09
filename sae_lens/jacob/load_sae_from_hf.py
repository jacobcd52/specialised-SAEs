import os
from huggingface_hub import hf_hub_download
from sae_lens.sae import SAE
import torch 
from sae_lens.config import DTYPE_MAP
def load_sae_from_hf(repo_id, filename, cfg_filename, device="cuda", dtype="float32"):
    # Make a directory to store the weights and cfg
    temp_gsae_path = "temp_sae"
    os.makedirs(temp_gsae_path, exist_ok=True)

    # Define the local paths for the files
    temp_weights_path = os.path.join(temp_gsae_path, "sae_weights.safetensors")
    temp_cfg_path = os.path.join(temp_gsae_path, "cfg.json")

    try:
        # Download weights
        print(f"Downloading weights from Hugging Face Hub")
        downloaded_weights_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_dir=temp_gsae_path
        )
        os.rename(downloaded_weights_path, temp_weights_path)
        print(f"GSAE weights file saved as {temp_weights_path}")

        # Download cfg
        print(f"Downloading cfg from Hugging Face Hub")
        downloaded_cfg_path = hf_hub_download(
            repo_id=repo_id, 
            filename=cfg_filename, 
            local_dir=temp_gsae_path
        )
        os.rename(downloaded_cfg_path, temp_cfg_path)
        print(f"GSAE cfg file saved as {temp_cfg_path}")
    except Exception as e:
        print(f"Error downloading weights or cfg: {e}")

    # Load weights into GSAE
    print(f"Loading weights into GSAE from {temp_weights_path}")                
    sae = SAE.load_from_pretrained(temp_gsae_path, device=device, dtype=dtype)
    return sae