import sys
sys.path.append("/root/specialised-SAEs/")
import os
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.jacob.load_sae_from_hf import load_sae_from_hf
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
import torch
from huggingface_hub import login, HfApi
login(token="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")
api = HfApi()

#########
dead_feature_window = 100_000
total_training_steps = 6000
batch_size = 4096

total_training_tokens = total_training_steps * batch_size
lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 20% of training

expansion_factor=2
model_name = "gemma-2-2b"

first_activation_pos = 2

d_in=2304 if model_name == "gemma-2-2b" else 2048

control_mixture = 0
lr = 1e-3

for (layer, gsae_id, gsae_width, l1_coefficient) in [
                                                    # (12, "layer_12/width_16k/average_l0_22", "16k", 20),
                                                     (7, "layer_7/width_16k/average_l0_20", "16k", 15),
                                                     (13, "layer_13/width_65k/average_l0_40", "65k", 40)
                                                     ]:
    for subject in ["hs_bio_cleaned", "hs_phys_cleaned", "hs_math_cleaned", "college_bio_cleaned", "college_phys_cleaned", "college_math_cleaned", "econ_cleaned", "history_cleaned"]:

        hook_name = f"blocks.{layer}.hook_resid_post" if model_name == "gemma-2-2b" else f"blocks.{layer}.hook_resid_pre"
        gsae_id_ = gsae_id.replace("/", "_")
        run_name = f"{model_name}_layer{layer}_{subject}_l1={l1_coefficient}_expansion={expansion_factor}_tokens={batch_size*total_training_steps}_gsae_id={gsae_id_}"
    
        cfg = LanguageModelSAERunnerConfig(
            # JACOB
            model_from_pretrained_kwargs = {"dtype" : "bfloat16"},
            b_dec_init_method="mean",
            # gsae_repo = 'jacobcd52/gemma2-gsae',
            # gsae_filename = 'sae_weights.safetensors',
            # gsae_cfg_filename = 'cfg.json',
            gsae_release = 'gemma-scope-2b-pt-res',
            gsae_id = gsae_id,

            first_activation_pos = first_activation_pos,
            
            apply_b_dec_to_input=True,  # We won't apply the decoder weights to the input.

            control_dataset_path=None,
            is_control_dataset_tokenized=False,
            control_mixture=control_mixture,

            save_final_checkpoint_locally = True,
            dtype="bfloat16",
            dataset_path=f'jacobcd52/{subject}',
            is_dataset_tokenized=False,
            wandb_project=f"{model_name}-layer{layer}-ssae",
            context_size=128,
            # from_pretrained_path="/root/specialised-SAEs/sae_lens/jacob/temp_sae",

            # Data Generating Function (Model + Training Distribution)
            architecture="gated",
            model_name=model_name,  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
            hook_name=hook_name,  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
            hook_layer=layer,  # Only one layer in the model.
            d_in=d_in,  # the width of the mlp output.
            streaming=True,  # we could pre-download the token dataset if it was small.
            # SAE Parameters
            mse_loss_normalization=None,  # We won't normalize the mse loss,
            expansion_factor=expansion_factor,  # the width of the SAE. Larger will result in better stats but slower training.
            normalize_sae_decoder=False,
            scale_sparsity_penalty_by_decoder_norm=False,
            decoder_heuristic_init=True,
            init_encoder_as_decoder_transpose=True,
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
            # Activation Store Parameters
            n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
            training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
            store_batch_size_prompts=32,
            # Resampling protocol
            use_ghost_grads=False,  # we don't use ghost grads anymore.
            feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
            dead_feature_window=dead_feature_window,  # would effect resampling or ghost grads if we were using it.
            dead_feature_threshold=1e-8,  # would effect resampling or ghost grads if we were using it.
            # WANDB
            log_to_wandb=True,  # always use wandb unless you are just testing code.
            run_name = run_name,
            wandb_log_frequency=20,
            eval_every_n_wandb_logs=20, #20
            # Misc
            device="cuda",
            seed=42,
            n_checkpoints=0,
            checkpoint_path=run_name,

        )
        ssae = SAETrainingRunner(cfg)
        ssae.run()

        # upload final checkpoint to HF
        final_checkpoint_path = None
        for path in os.listdir(cfg.checkpoint_path):
            if 'final' in path:
                final_checkpoint_path = cfg.checkpoint_path + "/" + path
            else:
                print("final checkpoint path not found")
        if final_checkpoint_path:
            cfg_path = final_checkpoint_path + "/cfg.json"
            sae_path = final_checkpoint_path + "/sae_weights.safetensors"

            # get descriptive file name

            name = run_name
            if cfg.control_mixture > 0:
                name += f"_control_mix={cfg.control_mixture}"

            # Define the repository ID
            repo_id = f"jacobcd52/{model_name}-ssae-{subject}"

            # Try to create the repository if it doesn't exist
            try:
                api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating the repository: {e}")
                # You might want to handle this error more gracefully

            # Upload the files to the repository
            try:
                api.upload_file(
                    path_or_fileobj=cfg_path,
                    path_in_repo=f"{name}_cfg.json",
                    repo_id=repo_id
                )

                api.upload_file(
                    path_or_fileobj=sae_path,
                    path_in_repo=f"{name}.safetensors",
                    repo_id=repo_id
                )
                print("Files uploaded successfully.")
            except Exception as e:
                print(f"An error occurred while uploading the files: {e}")
        else:
            print("saving failed - no final checkpoint found!")
