#%%
%load_ext autoreload
%autoreload 2
import os
import sys
sys.path.append("/root/specialised-SAEs/")
from huggingface_hub import hf_hub_download
from sae_lens.sae import SAE
from sae_lens.training.training_sae import TrainingSAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner
from tqdm import tqdm


#%%
# callum imports 
from IPython import get_ipython # type: ignore
ipython = get_ipython(); assert ipython is not None

# Standard imports
import torch
from datasets import load_dataset
import webbrowser
import os
from transformer_lens import utils, HookedTransformer
from datasets.arrow_dataset import Dataset
from huggingface_hub import hf_hub_download
import time

# Library imports

from sae_vis.utils_fns import get_device
from sae_vis_vedang.model_fns import AutoEncoder
from sae_vis_vedang.data_storing_fns import SaeVisData
from sae_vis_vedang.data_config_classes import SaeVisConfig

#%%
def load_sae_from_hf(repo_id, filename_no_suffix, device):
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
            filename=f"{filename_no_suffix}.safetensors", 
            local_dir=temp_gsae_path
        )
        os.rename(downloaded_weights_path, temp_weights_path)
        print(f"SAE weights file saved as {temp_weights_path}")

        # Download cfg
        print(f"Downloading cfg from Hugging Face Hub")
        downloaded_cfg_path = hf_hub_download(
            repo_id=repo_id, 
            filename=f"{filename_no_suffix}_cfg.json", 
            local_dir=temp_gsae_path
        )
        os.rename(downloaded_cfg_path, temp_cfg_path)
        print(f"SAE cfg file saved as {temp_cfg_path}")
    except Exception as e:
        print(f"Error downloading weights or cfg: {e}")

    # Load weights into SAE
    print(f"Loading weights into SAE from {temp_weights_path}")                
    sae = SAE.load_from_pretrained(temp_gsae_path, device=device)
    return sae

#%%

def get_tokens(
    activation_store: ActivationsStore,
    n_batches_to_sample_from: int = 2**10,
    n_prompts_to_select: int = 4096 * 6,
    control_mixture: float = 0.5
):
    all_tokens_list = []

    print("getting control tokens")
    pbar = tqdm(range(int(control_mixture*n_batches_to_sample_from)))
    for _ in pbar:
        batch_tokens = activation_store.get_control_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    print("getting specialised tokens")
    pbar = tqdm(range(int((1-control_mixture)*n_batches_to_sample_from)))
    for _ in pbar:
        batch_tokens = activation_store.get_batch_tokens()
        batch_tokens = batch_tokens[torch.randperm(batch_tokens.shape[0])][
            : batch_tokens.shape[0]
        ]
        all_tokens_list.append(batch_tokens)

    all_tokens = torch.cat(all_tokens_list, dim=0)
    all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]
    return all_tokens[:n_prompts_to_select]

#%%
total_training_steps = 3_000 
batch_size = 4096*2
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

control_mixture = 0.9
lr = 1e-3
l1_coefficient = 1
expansion_factor = 1

cfg = LanguageModelSAERunnerConfig(
    # JACOB
    gsae_repo = "jacobcd52/mats-saes",
    gsae_filename_no_suffix= "gpt2_resid_8_gated_gsae",
    is_control_dataset_tokenized=True,
    control_mixture=control_mixture,
    control_dataset_path="NeelNanda/openwebtext-tokenized-9b" if control_mixture > 0 else None,

    dataset_path="jacobcd52/physics-papers",
    is_dataset_tokenized=False,

    # Data Generating Function (Model + Training Distribution)
    architecture="standard",  # we'll use the gated variant.
    model_name="gpt2-small",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name="blocks.8.hook_resid_pre",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
    hook_layer=8,  # Only one layer in the model.
    d_in=768,  # the width of the mlp output.
    streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor=expansion_factor,  # the width of the SAE. Larger will result in better stats but slower training.
    b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
    apply_b_dec_to_input=True,  # We won't apply the decoder weights to the input.
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=False,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    # normalize_activations=False, JACOB
    # Training Parameters
    lr=lr,  # lower the better, we'll go fairly high to speed up the tutorial.
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=l1_coefficient,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=256,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project="scratch-ssae-stuff",
    run_name = f"l1={l1_coefficient}_expansion={expansion_factor}_control_mix={control_mixture}_tokens={batch_size*total_training_steps}_lr={lr}",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device="cuda",
    seed=42,
    n_checkpoints=1,
    checkpoint_path=f"phys_gpt2_ssae_l1_coeff={l1_coefficient}_expansion={expansion_factor}_control_mixture={control_mixture}_tokens={batch_size*total_training_steps}_lr={lr}",
    dtype="float32"
)
print("instantiating ssae")
ssae = SAETrainingRunner(cfg)
model = ssae.model
activation_store = ssae.activations_store
all_tokens_gpt = get_tokens(activation_store, control_mixture=1.0, n_batches_to_sample_from = 2**10)

#%%
sae = load_sae_from_hf("jacobcd52/mats-saes", "specialisedgpt2-small_l1_coeff=6.0_expansion=2_tokens=24576000_lr=0.001physics-papers_control_mix=0.0", "cuda")

#%%

torch.cuda.empty_cache()
import gc
gc.collect()

test_feature_idx_gpt = [i for i in range(50)]

feature_vis_config_gpt = SaeVisConfig(
    hook_point = sae.cfg.hook_name,
    features = test_feature_idx_gpt,
    batch_size = 8192,
    verbose = True,
)

sae_vis_data_gpt = SaeVisData.create(
    encoder = sae,
    model = model,
    tokens = all_tokens_gpt, # type: ignore
    cfg = feature_vis_config_gpt,
)

filename = "phys_features_owt.html"
sae_vis_data_gpt.save_feature_centric_vis(filename)

# %%
