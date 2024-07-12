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


# percent of params that are SSAE = (ssae_expansion/4) / (n_layers)
# percent of params that are GSAE = (gsae_expansion/4) / (n_layers)

# SSAE
# in units of model FPs, a training step requires following compute:
# 1 + 1*(percent_gsae) + 2*(percent_ssae) = 1 + (2*ssae_expansion + gsae_expansion)/(4*n_layers)

# GSAE-finetune
# training step compute:
# 1 + 2*(percent_gsae) = 1 + gsae_expansion/(2*n_layers)


# so GSAE_compute = SSAE_compute + gsae_expansion/(2*n_layers)



total_training_steps = 5000//16
batch_size = 8192*4
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

expansion_factor=16

for lr in [1e-5]:
    for l1_coefficient in [2, 5, 10]:
        for control_mixture in [0.5, 0.1, 0.0]:
        
            cfg = LanguageModelSAERunnerConfig(
                # JACOB
                model_from_pretrained_kwargs = {"dtype" : "bfloat16"},
                b_dec_init_method="geometric_median",
                # gsae_repo = 'jacobcd52/gemma2-gsae',
                # gsae_filename = 'sae_weights.safetensors',
                # gsae_cfg_filename = 'cfg.json',
                control_dataset_path="Skylion007/openwebtext" if control_mixture > 0 else None,
                is_control_dataset_tokenized=False,
                control_mixture=control_mixture,
                save_final_checkpoint_locally = True,
                dtype="bfloat16",
                dataset_path='jacobcd52/physics-papers',
                is_dataset_tokenized=False,
                wandb_project="gemma2-gsae-finetune-phys",
                context_size=128,
                from_pretrained_path="/root/specialised-SAEs/sae_lens/jacob/temp_sae",

                # Data Generating Function (Model + Training Distribution)
                architecture="gated",  # we'll use the gated variant.
                model_name="gemma-2b-it",  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
                hook_name="blocks.12.hook_resid_pre",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
                hook_layer=12,  # Only one layer in the model.
                d_in=2048,  # the width of the mlp output.
                streaming=True,  # we could pre-download the token dataset if it was small.
                # SAE Parameters
                mse_loss_normalization=None,  # We won't normalize the mse loss,
                expansion_factor=expansion_factor,  # the width of the SAE. Larger will result in better stats but slower training.
                apply_b_dec_to_input=True,  # We won't apply the decoder weights to the input.
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
                dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
                dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
                # WANDB
                log_to_wandb=True,  # always use wandb unless you are just testing code.
                run_name = f"l1={l1_coefficient}_expansion={expansion_factor}_control_mix={control_mixture}_tokens={batch_size*total_training_steps}_lr={lr}",
                wandb_log_frequency=30, #30
                eval_every_n_wandb_logs=20, #20
                # Misc
                device="cuda",
                seed=42,
                n_checkpoints=0,
                checkpoint_path=f"l1_coeff={l1_coefficient}_expansion={expansion_factor}_tokens={batch_size*total_training_steps}_lr={lr}",

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

                name = f"l1_coeff={l1_coefficient}_tokens={total_training_tokens}_lr={lr}"
                if cfg.control_mixture > 0:
                    name += f"_control_mix={cfg.control_mixture}"

                # Upload the files to a new repository
                api.upload_file(
                    path_or_fileobj=cfg_path,
                    path_in_repo = name + "_cfg.json",
                    repo_id="jacobcd52/gemma2-gsae-finetune-phys"
                )

                api.upload_file(
                    path_or_fileobj=sae_path,
                    path_in_repo = name + ".safetensors",
                    repo_id="jacobcd52/gemma2-gsae-finetune-phys"
                )
            else:
                print("saving failed - no final checkpoint found!")
