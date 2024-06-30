import sys
sys.path.append("/root/specialised-SAEs/")
import os
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.training.training_sae import TrainingSAEConfig, TrainingSAE
from sae_lens.sae import SAE
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.activations_store import ActivationsStore
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from huggingface_hub import login, HfApi
login(token="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")
api = HfApi()




total_training_steps = 5_000 
batch_size = 4096*2
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

expansion_factor = 4


for lr in [1e-4, 1e-5]:
    for l1_coefficient in [30, 40]:
        for control_mixture in [0, 0.1, 0.5]:
            cfg = LanguageModelSAERunnerConfig(
                # JACOB
                gsae_repo = "jacobcd52/mats-saes",
                gsae_filename_no_suffix= "gpt2_resid_8_gated_gsae",
                control_dataset_path="NeelNanda/openwebtext-tokenized-9b",
                is_control_dataset_tokenized=True,
                control_mixture=control_mixture,

                dataset_path="jacobcd52/physics-papers",
                is_dataset_tokenized=False,

                # Data Generating Function (Model + Training Distribution)
                architecture="gated",  # we'll use the gated variant.
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
                lr=5e-5,  # lower the better, we'll go fairly high to speed up the tutorial.
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
                wandb_project="physics-SSAE-gpt2",
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
            print("finished instantiating ssae")
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

                name = f"{cfg.model_name}_l1_coeff={l1_coefficient}_expansion={expansion_factor}_tokens={batch_size*total_training_steps}_lr={lr}"
                if cfg.gsae_repo:
                    name = "specialised" + name + f"{cfg.dataset_path.split('/')[-1]}_control_mix={cfg.control_mixture}"

                # Upload the files to a new repository
                api.upload_file(
                    path_or_fileobj=cfg_path,
                    path_in_repo = name + "_cfg.json",
                    repo_id="jacobcd52/mats-saes"
                )

                api.upload_file(
                    path_or_fileobj=sae_path,
                    path_in_repo = name + ".safetensors",
                    repo_id="jacobcd52/mats-saes"
                )
            else:
                print("saving failed - no final checkpoint found!")
