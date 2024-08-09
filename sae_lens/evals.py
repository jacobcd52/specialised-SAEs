import argparse
import re
from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping

import einops
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule

from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activations_store import ActivationsStore


# Everything by default is false so the user can just set the ones they want to true
@dataclass
class EvalConfig:
    batch_size_prompts: int | None = None

    # Reconstruction metrics
    n_eval_reconstruction_batches: int = 10
    compute_kl: bool = False
    compute_ce_loss: bool = False

    # Sparsity and variance metrics
    n_eval_sparsity_variance_batches: int = 1
    compute_l2_norms: bool = False
    compute_sparsity_metrics: bool = False
    compute_variance_metrics: bool = False


def get_eval_everything_config(
    batch_size_prompts: int | None = None,
    n_eval_reconstruction_batches: int = 10,
    n_eval_sparsity_variance_batches: int = 1,
) -> EvalConfig:
    """
    Returns an EvalConfig object with all metrics set to True, so that when passed to run_evals all available metrics will be run.
    """
    return EvalConfig(
        batch_size_prompts=batch_size_prompts,
        n_eval_reconstruction_batches=n_eval_reconstruction_batches,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
    )


@torch.no_grad()
def run_evals(
    sae: SAE,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    eval_config: EvalConfig = EvalConfig(),
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:

    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index
    ### Evals

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    # Get Reconstruction Score
    losses_df = recons_loss_batched(
        sae,
        model,
        activation_store,
        n_batches=n_eval_batches,
        eval_batch_size_prompts=eval_batch_size_prompts,
    )

    main_loss = losses_df["main_loss"].mean()
    main_recons_loss = losses_df["main_recons_loss"].mean()
    main_zero_abl_loss = losses_df["main_zero_abl_loss"].mean()
    control_loss = losses_df["control_loss"].mean()
    control_recons_loss = losses_df["control_recons_loss"].mean()
    control_zero_abl_loss = losses_df["control_zero_abl_loss"].mean()
    main_score = losses_df["main_score"].mean()
    control_score = losses_df["control_score"].mean()

    metrics = {
        # CE Loss
        "metrics/main_loss" : main_loss,
        "metrics/control_loss" : control_loss,
        "metrics/control_recons_loss": control_recons_loss,
        "metrics/main_recons_loss": main_recons_loss,
        "metrics/control_zero_abl_loss": control_zero_abl_loss,
        "metrics/main_zero_abl_loss": main_zero_abl_loss,
        "metrics/main_score": main_score,
        "metrics/control_score": control_score,
    }

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    return metrics


def recons_loss_batched(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int = 100,
    eval_batch_size_prompts: int | None = None,
):
    losses = []
    for _ in range(n_batches):
        main_batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        main_loss, main_recons_loss, main_zero_abl_loss, main_score = get_recons_loss(
            sae,
            model,
            main_batch_tokens,
            activation_store,
        )

        if activation_store.control_dataset:
            control_batch_tokens = activation_store.get_control_batch_tokens(eval_batch_size_prompts)
            control_loss, control_recons_loss, control_zero_abl_loss, control_score = get_recons_loss(
                sae,
                model,
                control_batch_tokens,
                activation_store,
            )
        else:
            control_loss = torch.tensor(0.0)
            control_recons_loss = torch.tensor(0.0)
            control_zero_abl_loss = torch.tensor(0.0)
            control_score = torch.tensor(0.0)

        losses.append(
            (
                main_loss.mean().item(),
                main_recons_loss.mean().item(),
                main_zero_abl_loss.mean().item(),
                main_score.mean().item(),
                control_loss.mean().item(),
                control_recons_loss.mean().item(), 
                control_zero_abl_loss.mean().item(),
                control_score.mean().item()

            )
        )

    losses = pd.DataFrame(
        losses, columns=cast(Any, ["main_loss", 
                                   "main_recons_loss", 
                                   "main_zero_abl_loss",
                                   "main_score",
                                   "control_loss", 
                                   "control_recons_loss", 
                                   "control_zero_abl_loss",
                                   "control_score"])
    )

    return losses


@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:
    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    model.reset_hooks()
    loss = model(batch_tokens, return_type="loss", **model_kwargs)

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass. JACOB
        if sae.gsae:
            activations = sae.decode(sae.encode(activations)).to(activations.dtype) + sae.gsae(activations).to(activations.dtype)
        else:
            activations = sae.decode(sae.encode(activations)).to(activations.dtype)
 
        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)

        return activations.to(original_device)

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(
            activations.dtype
        )

        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        return new_activations.to(original_device)

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        new_activations = sae.decode(sae.encode(activations[:, :, head_index])).to(
        new_activations = sae.decode(sae.encode(activations[:, :, head_index])).to(
            activations.dtype
        )
        activations[:, :, head_index] = new_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)
        return activations.to(original_device)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)

        # JACOB
        if sae.gsae:
            activations = sae.gsae(activations)
        else:
            activations = torch.zeros_like(activations)

        return activations.to(original_device)

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook

    model.reset_hooks()
    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        **model_kwargs,
    )

    model.reset_hooks()
    zero_abl_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        **model_kwargs,
    )

    div_val = zero_abl_loss - loss
    div_val[torch.abs(div_val) < 0.0001] = 1.0
    score = (zero_abl_loss - recons_loss) / div_val

    return loss, recons_loss, zero_abl_loss, score
