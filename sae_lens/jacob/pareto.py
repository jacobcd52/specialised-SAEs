import sys
sys.path.append("/root/specialised-SAEs")
from datasets import load_dataset
from transformer_lens import utils, HookedTransformer
import gc
import torch
from sae_lens.sae import SAE
from load_sae_from_hf import load_sae_from_hf
from analysis_fns import get_owt_and_spec_tokens, get_l0_freqs_loss_fvu, sweep, get_freq_plots, get_cossim_plots
from sae_lens.config import DTYPE_MAP
from tqdm import tqdm
import pandas as pd
import plotly.express as px
from huggingface_hub import login, HfApi, create_repo
import matplotlib.pyplot as plt

login(token="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")
api = HfApi()

torch.set_grad_enabled(False)
DTYPE = "bfloat16"




# Load SAEs, model and dataset
gsae = load_sae_from_hf("jacobcd52/gemma2-gsae", 
                        "sae_weights.safetensors", 
                        "cfg.json",
                        device="cuda", dtype=DTYPE)
model = HookedTransformer.from_pretrained_no_processing("gemma-2b-it", device="cuda", dtype=DTYPE)

# Sanity check: the GSAE error should be smaller than the original activation
loss, cache = model.run_with_cache("My name is Jacob, and I come from London, England.", return_type="loss", names_filter=[gsae.cfg.hook_name])
act = cache[gsae.cfg.hook_name]
print("\n\nSanity check:")
print(f"GSAE error norm = {(gsae(act) - act)[:,1:].norm().item():.1f}")
print(f"Input act norm = {act.norm().item():.1f}")



########### SET THESE #############
ssae_l1_list = [20, 10, 6]
direct_sae_l1_list =  [5, 2, 1]
gsae_ft_l1_list = [4, 2, 1]
###################################

def load_saes_and_data(subject):
    ssae_list = [load_sae_from_hf(f"jacobcd52/gemma-2b-it-ssae-{subject}",
                            f"gemma-2b-it_layer12_{subject}_l1={l1}_expansion=2_tokens=8192000_gsae_id=layer_12_stepan.safetensors",
                            f"gemma-2b-it_layer12_{subject}_l1={l1}_expansion=2_tokens=8192000_gsae_id=layer_12_stepan_cfg.json",
                            device="cuda", dtype=DTYPE)
                for l1 in ssae_l1_list]

    direct_sae_list = [load_sae_from_hf(f"jacobcd52/gemma-2b-it-directsae-{subject}",
                                        f"gemma-2b-it_layer12_{subject}_l1={l1}_expansion=2_tokens=8192000.safetensors",
                                        f"gemma-2b-it_layer12_{subject}_l1={l1}_expansion=2_tokens=8192000_cfg.json",
                                        device="cuda", dtype=DTYPE)
                for l1 in direct_sae_l1_list]

    gsae_ft_list = [load_sae_from_hf(f"jacobcd52/gemma-2b-it-gsae-ft-{subject}",
                                    f"gsaefinetune_gemma-2b-it_layer12_{subject}_l1={l1}_expansion=16_tokens=8192000.safetensors",
                                    f"gsaefinetune_gemma-2b-it_layer12_{subject}_l1={l1}_expansion=16_tokens=8192000_cfg.json",
                                    device="cuda", dtype=DTYPE)
                for l1 in gsae_ft_l1_list]
    for sae in gsae_ft_list:
        sae.cfg.apply_b_dec_to_input = False

    all_ctx_lengths = [gsae.cfg.context_size] + [sae.cfg.context_size for sae in ssae_list + direct_sae_list + gsae_ft_list]
    ctx_length = min(all_ctx_lengths)
    print("context length =", ctx_length)

    all_hook_pts = set([gsae.cfg.hook_name] + [sae.cfg.hook_name for sae in ssae_list + gsae_ft_list])
    assert len(all_hook_pts) == 1, "All models must have the same hook point"
    hook_pt = all_hook_pts.pop()
    print("hook point =", hook_pt)

    owt_tokens, spec_tokens = get_owt_and_spec_tokens(model, f"jacobcd52/{subject}", ctx_length=ctx_length)

    return ssae_list, direct_sae_list, gsae_ft_list, ctx_length, hook_pt, owt_tokens, spec_tokens




def run_subject(model, subject, num_tokens=100_000):
    ssae_list, direct_sae_list, gsae_ft_list, ctx_length, hook_pt, owt_tokens, spec_tokens = load_saes_and_data(subject)

    # Get floor and ceiling losses (i.e. clean and with GSAE)
    print(f"\n\ngetting floor and ceiling losses for {subject}\n")
    _, _, clean_owt_losses, _  = get_l0_freqs_loss_fvu(model, "clean", owt_tokens, num_tokens=num_tokens)
    _, _, clean_spec_losses, _ = get_l0_freqs_loss_fvu(model, "clean", spec_tokens, num_tokens=num_tokens)

    gsae_owt_l0, gsae_owt_freqs, gsae_owt_losses, gsae_owt_fvu  = get_l0_freqs_loss_fvu(model, [gsae], owt_tokens, num_tokens=num_tokens)
    gsae_spec_l0, gsae_spec_freqs, gsae_spec_losses, gsae_spec_fvu = get_l0_freqs_loss_fvu(model, [gsae], spec_tokens, num_tokens=num_tokens)   

    print(f"clean owt loss = {clean_owt_losses.mean().item():.3f}")
    print(f"gsae owt loss = {gsae_owt_losses.mean().item():.3f}")
    print(f"gsae owt L0 = {gsae_owt_l0:.1f}")
    print(f"gsae owt FVU = {gsae_owt_fvu:.2f}")

    print(f"\nclean spec loss = {clean_spec_losses.mean().item():.3f}")
    print(f"gsae spec loss = {gsae_spec_losses.mean().item():.3f}")
    print(f"gsae spec L0 = {gsae_spec_l0:.1f}")
    print(f"gsae spec FVU = {gsae_spec_fvu:.2f}")

    print(f"\n\ngetting pareto data for {subject}\n")
    # get pareto data for SSAEs
    ssae_owt_l0, ssae_owt_freqs, ssae_owt_scores, ssae_owt_fvu_recovered = sweep(model, [[gsae, ssae] for ssae in ssae_list], owt_tokens, gsae_owt_losses, clean_owt_losses, gsae_owt_fvu, num_tokens=num_tokens)
    ssae_spec_l0, ssae_spec_freqs, ssae_spec_scores, ssae_spec_fvu_recovered = sweep(model, [[gsae, ssae] for ssae in ssae_list], spec_tokens, gsae_spec_losses, clean_spec_losses, gsae_spec_fvu, num_tokens=num_tokens)

    # get pareto data for GSAE finetunes
    gsae_ft_owt_l0, gsae_ft_owt_freqs, gsae_ft_owt_scores, gsae_ft_owt_fvu_recovered = sweep(model, [[gsae] for gsae in gsae_ft_list], owt_tokens, gsae_owt_losses, clean_owt_losses, gsae_owt_fvu, num_tokens=num_tokens)
    gsae_ft_spec_l0, gsae_ft_spec_freqs, gsae_ft_spec_scores, gsae_ft_spec_fvu_recovered = sweep(model, [[gsae] for gsae in gsae_ft_list], spec_tokens, gsae_spec_losses, clean_spec_losses, gsae_spec_fvu, num_tokens=num_tokens)

    # get pareto data for direct SAEs
    direct_owt_l0, direct_owt_freqs, direct_owt_scores, direct_owt_fvu_recovered = sweep(model, [[sae] for sae in direct_sae_list], owt_tokens, gsae_owt_losses, clean_owt_losses, gsae_owt_fvu, num_tokens=num_tokens)
    direct_spec_l0, direct_spec_freqs, direct_spec_scores, direct_spec_fvu_recovered = sweep(model, [[sae] for sae in direct_sae_list], spec_tokens, gsae_spec_losses, clean_spec_losses, gsae_spec_fvu, num_tokens=num_tokens)

    get_freq_plots(ssae_owt_freqs, direct_owt_freqs, gsae_ft_owt_freqs,
                    ssae_spec_freqs, direct_spec_freqs, gsae_ft_spec_freqs, 
                    subject)

    get_cossim_plots(gsae, gsae_ft_list, ssae_list, 
                        ssae_l1_list, gsae_ft_l1_list,                    
                        subject)

    # return pareto data
    gsae_ft_owt_data =  {"l1": gsae_ft_l1_list, 
                        "l0": gsae_ft_owt_l0, 
                        "fvu": gsae_ft_owt_fvu_recovered, 
                        "scores": gsae_ft_owt_scores}
    ssae_owt_data = {"l1": ssae_l1_list,
                    "l0": ssae_owt_l0,
                    "fvu": ssae_owt_fvu_recovered,
                    "scores": ssae_owt_scores}
    direct_owt_data = {"l1": direct_sae_l1_list,
                    "l0": direct_owt_l0,
                    "fvu": direct_owt_fvu_recovered,
                    "scores": direct_owt_scores}
    gsae_ft_spec_data = {"l1": gsae_ft_l1_list,
                        "l0": gsae_ft_spec_l0,
                        "fvu": gsae_ft_spec_fvu_recovered,
                        "scores": gsae_ft_spec_scores}
    ssae_spec_data = {"l1": ssae_l1_list,
                    "l0": ssae_spec_l0,
                    "fvu": ssae_spec_fvu_recovered,
                    "scores": ssae_spec_scores}
    direct_spec_data = {"l1": direct_sae_l1_list,
                    "l0": direct_spec_l0,
                    "fvu": direct_spec_fvu_recovered,
                    "scores": direct_spec_scores}

    return gsae_ft_owt_data, ssae_owt_data, direct_owt_data, gsae_ft_spec_data, ssae_spec_data, direct_spec_data




# RUN
subject_to_data = {}
for subject in ["hs_bio_cleaned",
                "hs_math_cleaned", "hs_phys_cleaned", 
                "college_bio_cleaned", "college_math_cleaned", "college_phys_cleaned",
                "history_cleaned", "econ_cleaned"
                ]:
    subject_to_data[subject] = run_subject(model, subject, num_tokens=1_000_000)

# Save as a pickle file
import pickle
with open('subject_to_data.pkl', 'wb') as pickle_file:
    pickle.dump(subject_to_data, pickle_file)