"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import json
import os
from dataclasses import dataclass, fields
from typing import Any, Optional
import requests

import einops
import torch
from jaxtyping import Float
from torch import nn
from safetensors import safe_open
import json
from huggingface_hub import hf_hub_download
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict




from sae_lens.config import LanguageModelSAERunnerConfig, DTYPE_MAP
from sae_lens.config import DTYPE_MAP, LanguageModelSAERunnerConfig
from sae_lens.sae import SAE, SAEConfig
from sae_lens.toolkit.pretrained_sae_loaders import (
    handle_config_defaulting,
    read_sae_from_disk,
)
from sae_lens.jacob.load_sae_from_hf import load_sae_from_hf

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    main_mse_loss: float
    control_mse_loss: float
    target_output_norm : float
    main_output_norm : float
    control_output_norm : float
    l1_loss: float
    ghost_grad_loss: float
    auxiliary_reconstruction_loss: float = 0.0


@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig):
    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    noise_scale: float
    decoder_orthogonal_init: bool
    mse_loss_normalization: Optional[str]
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False

    # JACOB
    first_activation_pos : Optional[int] = None
    gsae_repo : Optional[str] = None
    gsae_filename : Optional[str] = None
    gsae_cfg_filename : Optional[str] = None
    gsae_release : Optional[str] = None
    gsae_id : Optional[str] = None
    control_dataset_path : Optional[str] = None
    control_mixture : float = 0.0
    is_control_dataset_tokenized : bool = True
    save_final_checkpoint_locally : bool = True

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":

        return cls(
            # JACOB
            first_activation_pos = cfg.first_activation_pos,
            gsae_repo = cfg.gsae_repo,
            gsae_filename = cfg.gsae_filename,
            gsae_cfg_filename = cfg.gsae_cfg_filename,
            gsae_release = cfg.gsae_release,
            gsae_id = cfg.gsae_id,
            control_dataset_path = cfg.control_dataset_path,
            control_mixture=cfg.control_mixture,
            is_control_dataset_tokenized = cfg.is_control_dataset_tokenized,
            save_final_checkpoint_locally = cfg.save_final_checkpoint_locally,

            # base config
            architecture=cfg.architecture,
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            activation_fn_str=cfg.activation_fn,
            activation_fn_kwargs=cfg.activation_fn_kwargs,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            finetuning_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            # Training cfg
            l1_coefficient=cfg.l1_coefficient,
            lp_norm=cfg.lp_norm,
            use_ghost_grads=cfg.use_ghost_grads,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            noise_scale=cfg.noise_scale,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            normalize_activations=cfg.normalize_activations,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }
        return TrainingSAEConfig(**valid_config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "noise_scale": self.noise_scale,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn_str": self.activation_fn_str,
            "activation_fn_kwargs": self.activation_fn_kwargs,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "device": self.device,
            "context_size": self.context_size,
            "prepend_bos": self.prepend_bos,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "normalize_activations": self.normalize_activations,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "sae_lens_training_version": self.sae_lens_training_version,

            # JACOB
            "first_activation_pos" : self.first_activation_pos,
            "gsae_repo" : self.gsae_repo,
            "gsae_filename" : self.gsae_filename,
            "gsae_cfg_filename" : self.gsae_cfg_filename,
            "gsae_release" : self.gsae_release,
            "gsae_id" : self.gsae_id,
            "control_dataset_path" : self.control_dataset_path,
            "control_mixture" : self.control_mixture,
            "is_control_dataset_tokenized" : self.is_control_dataset_tokenized,
            "save_final_checkpoint_locally" : self.save_final_checkpoint_locally
        }


from huggingface_hub import hf_hub_download

class TrainingSAE(SAE):
    """
    A SAE used for training. This class provides a `training_forward_pass` method which calculates
    losses used for training.
    """

    cfg: TrainingSAEConfig
    use_error_term: bool
    dtype: torch.dtype
    device: torch.device

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):

        base_sae_cfg = SAEConfig.from_dict(cfg.get_base_sae_cfg_dict())
        super().__init__(base_sae_cfg)
        self.cfg = cfg  # type: ignore

        # JACOB
        self.load_gsae()
        #

        self.encode_with_hidden_pre_fn = (
            self.encode_with_hidden_pre
            if cfg.architecture != "gated"
            else self.encode_with_hidden_pre_gated
        )

        self.check_cfg_compatibility()

        self.use_error_term = use_error_term

        self.initialize_weights_complex()

        # The training SAE will assume that the activation store handles
        # reshaping.
        self.turn_off_forward_pass_hook_z_reshaping()

        self.mse_loss_fn = self._get_mse_loss_fn()

    def load_gsae(self):
        # JACOB get gsae from huggingface
        self.gsae = None

        if self.cfg.gsae_repo:
            self.gsae = load_sae_from_hf(self.cfg.gsae_repo, 
                                         self.cfg.gsae_filename, 
                                         self.cfg.gsae_cfg_filename, 
                                         device=self.cfg.device,
                                         dtype=self.cfg.dtype)
    
        elif self.cfg.gsae_release:
            self.gsae, _, _ = SAE.from_pretrained(
                release = self.cfg.gsae_release,
                sae_id = self.cfg.gsae_id, # won't always be a hook point
                device = self.cfg.device,
                dtype=self.cfg.dtype
            )

        if self.gsae:
            for param in self.gsae.parameters():
                param.requires_grad = False
            print()
            print("self.cfg.dtype", self.cfg.dtype)
            print("gsae loaded, dtype", self.gsae.W_dec.dtype)
            print()
            assert(self.gsae.use_error_term == False)
            assert self.gsae.cfg.hook_name == self.cfg.hook_name, f"hook_name mismatch: {self.gsae.cfg.hook_name} vs {self.cfg.hook_name}"
            assert self.gsae.cfg.model_name == self.cfg.model_name, f"model_name mismatch: {self.gsae.cfg.model_name} vs {self.cfg.model_name}"
        else:
            print()
            print("no gsae loaded")
            print()    

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAE":
        return cls(TrainingSAEConfig.from_dict(config_dict))

    def check_cfg_compatibility(self):
        if self.cfg.architecture == "gated":
            assert (
                self.cfg.use_ghost_grads is False
            ), "Gated SAEs do not support ghost grads"
            assert self.use_error_term is False, "Gated SAEs do not support error terms"

    def encode_standard(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calcuate SAE features from inputs
        """
        feature_acts, _ = self.encode_with_hidden_pre_fn(x)
        return feature_acts

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:

        x = x.to(self.dtype)
        x = self.reshape_fn_in(x)  # type: ignore
        x = self.hook_sae_input(x)
        x = self.run_time_activation_norm_fn_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        hidden_pre_noised = hidden_pre + (
            torch.randn_like(hidden_pre) * self.cfg.noise_scale * self.training
        )
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))
        return feature_acts, hidden_pre_noised

    def encode_with_hidden_pre_gated(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:

        x = x.to(self.dtype)
        x = self.reshape_fn_in(x)  # type: ignore
        x = self.hook_sae_input(x)
        x = self.run_time_activation_norm_fn_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)

        # Gating path with Heaviside step function
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        # magnitude_pre_activation_noised = magnitude_pre_activation + (
        #     torch.randn_like(magnitude_pre_activation) * self.cfg.noise_scale * self.training
        # )
        feature_magnitudes = self.activation_fn(
            magnitude_pre_activation
        )  # magnitude_pre_activation_noised)
        # Return both the gated feature activations and the magnitude pre-activations
        return (
            active_features * feature_magnitudes,
            magnitude_pre_activation,
        )  # magnitude_pre_activation_noised

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"],
    ) -> Float[torch.Tensor, "... d_in"]:

        feature_acts, _ = self.encode_with_hidden_pre_fn(x)
        sae_out = self.decode(feature_acts)

        return sae_out

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:

        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        feature_acts, _ = self.encode_with_hidden_pre_fn(sae_in)
        sae_out = self.decode(feature_acts)
        # JACOB
        if self.gsae:
            assert len(sae_in.shape) == 2 # expect [batch d_model]
            control_batch_size = int(sae_in.size(0) * self.cfg.control_mixture)
            # get sum of main MSE loss and control loss
            target = sae_in - self.gsae(sae_in) 
            # if we're training a GSAE, we want the SSAE to output 0 on the control dataset
            # otherwise, we just demand usual reconstruction on both control and main datasets
            if self.gsae:
                target[:control_batch_size] = 0
            per_item_mse_loss = self.mse_loss_fn(sae_out, target)

            # calculate control and main losses for logging
            mse_loss = per_item_mse_loss.sum(dim=-1).mean()  # chop off BOS
            control_mse_loss = per_item_mse_loss[:control_batch_size].sum(dim=-1).mean()
            main_mse_loss = per_item_mse_loss[control_batch_size:].sum(dim=-1).mean()

            # calculate norms for logging
            target_output_norm = torch.norm(target[control_batch_size:], dim=-1).mean()
            main_output_norm = torch.norm(sae_out[control_batch_size:], dim=-1).mean()
            control_output_norm = torch.norm(sae_out[:control_batch_size], dim=-1).mean()

        else:
            target = sae_in
            per_item_mse_loss = self.mse_loss_fn(sae_out, target)
            mse_loss = per_item_mse_loss.sum(dim=-1).mean()
            control_mse_loss = torch.tensor(0.0)
            main_mse_loss = mse_loss
            target_output_norm = target.norm(dim=-1).mean()
            main_output_norm = torch.norm(sae_out, dim=-1).mean()
            control_output_norm = torch.tensor(0.0)


        # GHOST GRADS
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None:

            # first half of second forward pass
            _, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
            ghost_grad_loss = self.calculate_ghost_grad_loss(
                x=sae_in,
                sae_out=sae_out,
                per_item_mse_loss=per_item_mse_loss,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
        else:
            ghost_grad_loss = 0.0

        if self.cfg.architecture == "gated":
            # Gated SAE Loss Calculation
            # Shared variables
            sae_in_centered = (
                self.reshape_fn_in(sae_in) - self.b_dec * self.cfg.apply_b_dec_to_input
            )
            pi_gate = sae_in_centered @ self.W_enc + self.b_gate
            pi_gate_act = torch.relu(pi_gate)

            # SFN sparsity loss - summed over the feature dimension and averaged over the batch
            l1_loss = (
                current_l1_coefficient
                * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
            )

            # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
            via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
            aux_reconstruction_loss = torch.sum(
                (via_gate_reconstruction - target) ** 2, dim=-1
            ).mean()

            loss = mse_loss + l1_loss + aux_reconstruction_loss
        else:
            # default SAE sparsity loss
            weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
            sparsity = weighted_feature_acts.norm(
                p=self.cfg.lp_norm, dim=-1
            )  # sum over the feature dimension

            l1_loss = (current_l1_coefficient * sparsity).mean()
            loss = mse_loss + l1_loss + ghost_grad_loss

            aux_reconstruction_loss = torch.tensor(0.0)

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            target_output_norm=target_output_norm.item(),
            main_output_norm=main_output_norm.item(),
            control_output_norm=control_output_norm.item(),
            loss=loss,
            control_mse_loss=control_mse_loss.item(),
            main_mse_loss=main_mse_loss.item(),
            l1_loss=l1_loss.item(),
            ghost_grad_loss=(
                ghost_grad_loss.item()
                if isinstance(ghost_grad_loss, torch.Tensor)
                else ghost_grad_loss
            ),
            auxiliary_reconstruction_loss=aux_reconstruction_loss.item(),
        )

    def calculate_ghost_grad_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        per_item_mse_loss: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
    ) -> torch.Tensor:

        # 1.
        residual = x - sae_out
        l2_norm_residual = torch.norm(residual, dim=-1)

        # 2.
        # ghost grads use an exponentional activation function, ignoring whatever
        # the activation function is in the SAE. The forward pass uses the dead neurons only.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

        # 3. There is some fairly complex rescaling here to make sure that the loss
        # is comparable to the original loss. This is because the ghost grads are
        # only calculated for the dead neurons, so we need to rescale the loss to
        # make sure that the loss is comparable to the original loss.
        # There have been methodological improvements that are not implemented here yet
        # see here: https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Improving_ghost_grads
        per_item_mse_loss_ghost_resid = self.mse_loss_fn(ghost_out, residual.detach())
        mse_rescaling_factor = (
            per_item_mse_loss / (per_item_mse_loss_ghost_resid + 1e-6)
        ).detach()
        per_item_mse_loss_ghost_resid = (
            mse_rescaling_factor * per_item_mse_loss_ghost_resid
        )

        return per_item_mse_loss_ghost_resid.mean()

    @torch.no_grad()
    def _get_mse_loss_fn(self) -> Any:

        def standard_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.mse_loss(preds, target, reduction="none")

        def batch_norm_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            target_centered = target - target.mean(dim=0, keepdim=True)
            normalization = target_centered.norm(dim=-1, keepdim=True)
            return torch.nn.functional.mse_loss(preds, target, reduction="none") / (
                normalization + 1e-6
            )

        if self.cfg.mse_loss_normalization == "dense_batch":
            return batch_norm_mse_loss_fn
        else:
            return standard_mse_loss_fn

    @classmethod
    def load_from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> "TrainingSAE":

        # get the config
        config_path = os.path.join(path, SAE_CFG_PATH)
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_PATH)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
            dtype=DTYPE_MAP[cfg_dict["dtype"]],
        )
        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    def initialize_weights_complex(self):
        """ """

        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.decoder_heuristic_init:
            self.W_dec = nn.Parameter(
                torch.rand(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
            self.initialize_decoder_norm_constant_norm()

        # if using a GSAE, the thing we're reconstructing has small norm, so make initialized weights smaller
        if self.gsae:
            self.W_dec.data /= 10
        

        # Then we initialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.cfg.d_in,
                        self.cfg.d_sae,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            )

        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm()

    ## Initialization Methods
    @torch.no_grad()
    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    ## Training Utils
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """
        A heuristic proceedure inspired by:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        # TODO: Parameterise this as a function of m and n

        # ensure W_dec norms at unit norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
