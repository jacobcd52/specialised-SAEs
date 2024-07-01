from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Literal, cast

import numpy as np
import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule
from einops import rearrange

from sae_lens.config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)
from sae_lens.sae import SAE

HfDataset = DatasetDict | Dataset | IterableDatasetDict | IterableDataset


# Import additional modules if required
from datasets import load_dataset
from typing import Union

class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    control_dataset: HfDataset
    cached_activations_path: str | None
    tokens_column: Literal["tokens", "input_ids", "text"]
    control_tokens_column: Literal["tokens", "input_ids", "text"]
    hook_name: str
    hook_layer: int
    hook_head_index: int | None
    _dataloader: Iterator[Any] | None = None
    _storage_buffer: torch.Tensor | None = None
    device: torch.device

    @classmethod
    def from_config(
        cls,
        model: HookedRootModule,
        cfg: LanguageModelSAERunnerConfig | CacheActivationsRunnerConfig,
        dataset: HfDataset | None = None,
    ) -> "ActivationsStore":
        cached_activations_path = cfg.cached_activations_path
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None

        return cls(
            model=model,
            dataset=dataset or cfg.dataset_path,
            control_dataset_path=getattr(cfg, 'control_dataset_path', None),
            control_mixture=getattr(cfg, 'control_mixture', 0.0),
            is_control_dataset_tokenized=getattr(cfg, 'is_control_dataset_tokenized', False),
            streaming=cfg.streaming,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=cfg.normalize_activations,
            device=torch.device(cfg.act_store_device),
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        )


    # @classmethod
    # def from_sae(
    #     cls,
    #     model: HookedRootModule,
    #     sae: SAE,
    #     streaming: bool = True,
    #     store_batch_size_prompts: int = 8,
    #     n_batches_in_buffer: int = 8,
    #     train_batch_size_tokens: int = 4096,
    #     total_tokens: int = 10**9,
    #     device: str = "cpu",
    # ) -> "ActivationsStore":

    #     return cls(
    #         model=model,
    #         dataset=sae.cfg.dataset_path,
    #         control_dataset=sae.cfg.control_dataset_path,
    #         control_mixture=sae.cfg.control_mixture,
    #         is_control_dataset_tokenized=sae.cfg.is_control_dataset_tokenized,
    #         d_in=sae.cfg.d_in,
    #         hook_name=sae.cfg.hook_name,
    #         hook_layer=sae.cfg.hook_layer,
    #         hook_head_index=sae.cfg.hook_head_index,
    #         context_size=sae.cfg.context_size,
    #         prepend_bos=sae.cfg.prepend_bos,
    #         streaming=streaming,
    #         store_batch_size_prompts=store_batch_size_prompts,
    #         train_batch_size_tokens=train_batch_size_tokens,
    #         n_batches_in_buffer=n_batches_in_buffer,
    #         total_training_tokens=total_tokens,
    #         normalize_activations=sae.cfg.normalize_activations,
    #         dataset_trust_remote_code=sae.cfg.dataset_trust_remote_code,
    #         dtype=sae.cfg.dtype,
    #         device=torch.device(device),
    #     )

    def __init__(
        self,
        model: HookedRootModule,
        dataset: HfDataset | str,
        control_dataset_path: str,
        control_mixture: float,
        is_control_dataset_tokenized: bool,
        streaming: bool,
        hook_name: str,
        hook_layer: int,
        hook_head_index: int | None,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        cached_activations_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        autocast_lm: bool = False,
        dataset_trust_remote_code: bool | None = None,
    ):
        self.model = model
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.dataset = (
            load_dataset(
                dataset,
                split="train",
                streaming=streaming,
                trust_remote_code=dataset_trust_remote_code,  # type: ignore
            )
            if isinstance(dataset, str)
            else dataset
        )
        self.control_dataset = (
            load_dataset(
                control_dataset_path,
                split="train",
                streaming=streaming,
                trust_remote_code=dataset_trust_remote_code,  # type: ignore
            )
            if control_dataset_path
            else None
        )
        if self.control_dataset:
            print("using control dataset")
        else:
            print("no control dataset!")
        self.control_mixture = control_mixture if control_dataset_path else 0.0
        self.is_control_dataset_tokenized = is_control_dataset_tokenized if control_dataset_path else False
        self.hook_name = hook_name
        self.hook_layer = hook_layer
        self.hook_head_index = hook_head_index
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.cached_activations_path = cached_activations_path
        self.autocast_lm = autocast_lm

        self.n_dataset_processed = 0
        self.iterable_dataset = iter(self.dataset)
        self.iterable_control_dataset = iter(self.control_dataset) if self.control_dataset else None

        self.estimated_norm_scaling_factor = 1.0

        # Check if main dataset is tokenized
        dataset_sample = next(self.iterable_dataset)
        if "tokens" in dataset_sample.keys():
            self.is_dataset_tokenized = True
            self.tokens_column = "tokens"
        elif "input_ids" in dataset_sample.keys():
            self.is_dataset_tokenized = True
            self.tokens_column = "input_ids"
        elif "text" in dataset_sample.keys():
            self.is_dataset_tokenized = False
            self.tokens_column = "text"
        else:
            raise ValueError(
                "Main dataset must have a 'tokens', 'input_ids', or 'text' column."
            )
        self.iterable_dataset = iter(self.dataset)  # Reset iterator after checking

        if self.control_dataset:
            control_dataset_sample = next(self.iterable_control_dataset)
            if "tokens" in control_dataset_sample.keys():
                self.control_tokens_column = "tokens"
            elif "input_ids" in control_dataset_sample.keys():
                self.control_tokens_column = "input_ids"
            elif "text" in control_dataset_sample.keys():
                self.control_tokens_column = "text"
            else:
                raise ValueError(
                    "Control dataset must have a 'tokens', 'input_ids', or 'text' column."
                )
            self.iterable_control_dataset = iter(self.control_dataset)  # Reset iterator

        self.check_cached_activations_against_config()




    def check_cached_activations_against_config(self):
        if self.cached_activations_path is not None:  # EDIT: load from multi-layer acts
            assert self.cached_activations_path is not None  # keep pyright happy
            # Sanity check: does the cache directory exist?
            assert os.path.exists(
                self.cached_activations_path
            ), f"Cache directory {self.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

            self.next_cache_idx = 0  # which file to open next
            self.next_idx_within_buffer = 0  # where to start reading from in that file

            # Check that we have enough data on disk
            first_buffer = self.load_buffer(
                f"{self.cached_activations_path}/{self.next_cache_idx}.safetensors"
            )

            buffer_size_on_disk = first_buffer.shape[0]
            n_buffers_on_disk = len(os.listdir(self.cached_activations_path))

            # Note: we're assuming all files have the same number of tokens
            # (which seems reasonable imo since that's what our script does)
            n_activations_on_disk = buffer_size_on_disk * n_buffers_on_disk
            assert (
                n_activations_on_disk >= self.total_training_tokens
            ), f"Only {n_activations_on_disk/1e6:.1f}M activations on disk, but total_training_tokens is {self.total_training_tokens/1e6:.1f}M."

    def apply_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return activations * self.estimated_norm_scaling_factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        return activations / self.estimated_norm_scaling_factor

    def get_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return (self.d_in**0.5) / activations.norm(dim=-1).mean()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):

        norms_per_batch = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            acts = self.next_batch()
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(self.d_in) / mean_norm

        return scaling_factor

    @property
    def storage_buffer(self) -> torch.Tensor:
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.n_batches_in_buffer // 2)

        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader

    def get_batch_tokens(self, batch_size: int | None = None):
        """
        Streams a batch of tokens from the main dataset.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        context_size = self.context_size
        device = self.device

        batch_tokens = torch.zeros(
            size=(0, context_size), device=device, dtype=torch.long, requires_grad=False
        )

        current_batch = []
        current_length = 0

        while batch_tokens.shape[0] < batch_size:
            tokens = self._get_next_dataset_tokens()
            token_len = tokens.shape[0]

            # TODO: Fix this so that we are limiting how many tokens we get from the same context.
            assert self.model.tokenizer is not None  # keep pyright happy
            while token_len > 0 and batch_tokens.shape[0] < batch_size:
                # Space left in the current batch
                space_left = context_size - current_length

                # If the current tokens fit entirely into the remaining space
                if token_len <= space_left:
                    current_batch.append(tokens[:token_len])
                    current_length += token_len
                    break

                else:
                    # Take as much as will fit
                    current_batch.append(tokens[:space_left])

                    # Remove used part, add BOS
                    tokens = tokens[space_left:]
                    token_len -= space_left

                    # only add BOS if it's not already the first token
                    if self.prepend_bos:
                        bos_token_id_tensor = torch.tensor(
                            [self.model.tokenizer.bos_token_id],
                            device=tokens.device,
                            dtype=torch.long,
                        )
                        if tokens[0] != bos_token_id_tensor:
                            tokens = torch.cat(
                                (
                                    bos_token_id_tensor,
                                    tokens,
                                ),
                                dim=0,
                            )
                            token_len += 1
                    current_length = context_size

                # If a batch is full, concatenate and move to next batch
                if current_length == context_size:
                    full_batch = torch.cat(current_batch, dim=0)
                    batch_tokens = torch.cat(
                        (batch_tokens, full_batch.unsqueeze(0)), dim=0
                    )
                    current_batch = []
                    current_length = 0

        return batch_tokens[:batch_size].to(self.model.W_E.device)


    def get_control_batch_tokens(self, batch_size: int | None = None):
        """
        Streams a batch of tokens from the control dataset.
        """
        if self.control_dataset is None:
            raise ValueError("Control dataset is not provided.")

        if not batch_size:
            batch_size = self.store_batch_size_prompts
        context_size = self.context_size
        device = self.device

        batch_tokens = torch.zeros(
            size=(0, context_size), device=device, dtype=torch.long, requires_grad=False
        )

        current_batch = []
        current_length = 0

        while batch_tokens.shape[0] < batch_size:
            tokens = self._get_next_control_dataset_tokens()
            token_len = tokens.shape[0]

            # TODO: Fix this so that we are limiting how many tokens we get from the same context.
            assert self.model.tokenizer is not None  # keep pyright happy
            while token_len > 0 and batch_tokens.shape[0] < batch_size:
                # Space left in the current batch
                space_left = context_size - current_length

                # If the current tokens fit entirely into the remaining space
                if token_len <= space_left:
                    current_batch.append(tokens[:token_len])
                    current_length += token_len
                    break

                else:
                    # Take as much as will fit
                    current_batch.append(tokens[:space_left])

                    # Remove used part, add BOS
                    tokens = tokens[space_left:]
                    token_len -= space_left

                    # only add BOS if it's not already the first token
                    if self.prepend_bos:
                        bos_token_id_tensor = torch.tensor(
                            [self.model.tokenizer.bos_token_id],
                            device=tokens.device,
                            dtype=torch.long,
                        )
                        if tokens[0] != bos_token_id_tensor:
                            tokens = torch.cat(
                                (
                                    bos_token_id_tensor,
                                    tokens,
                                ),
                                dim=0,
                            )
                            token_len += 1
                    current_length = context_size

                # If a batch is full, concatenate and move to next batch
                if current_length == context_size:
                    full_batch = torch.cat(current_batch, dim=0)
                    batch_tokens = torch.cat(
                        (batch_tokens, full_batch.unsqueeze(0)), dim=0
                    )
                    current_batch = []
                    current_length = 0

        return batch_tokens[:batch_size].to(self.model.W_E.device)


    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor):
        """
        Returns activations of shape (batches, context, num_layers, d_in)

        d_in may result from a concatenated head dimension.
        """

        # Setup autocast if using
        if self.autocast_lm:
            autocast_if_enabled = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.autocast_lm,
            )
        else:
            autocast_if_enabled = contextlib.nullcontext()

        with autocast_if_enabled:
            layerwise_activations = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_name],
                stop_at_layer=self.hook_layer + 1,
                prepend_bos=self.prepend_bos,
                **self.model_kwargs,
            )[1]

        n_batches, n_context = batch_tokens.shape

        stacked_activations = torch.zeros((n_batches, n_context, 1, self.d_in))

        if self.hook_head_index is not None:
            stacked_activations[:, :, 0] = layerwise_activations[self.hook_name][
                :, :, self.hook_head_index
            ]
        elif (
            layerwise_activations[self.hook_name].ndim > 3
        ):  # if we have a head dimension
            stacked_activations[:, :, 0] = layerwise_activations[self.hook_name].view(
                n_batches, n_context, -1
            )
        else:
            stacked_activations[:, :, 0] = layerwise_activations[self.hook_name]

        return stacked_activations

    @torch.no_grad()
    def get_buffer(self, n_batches_in_buffer: int) -> tuple[torch.Tensor, torch.Tensor]:
        context_size = self.context_size
        batch_size = self.store_batch_size_prompts
        d_in = self.d_in
        num_layers = 1
        total_tokens = batch_size * n_batches_in_buffer * context_size

        if self.cached_activations_path is not None:
            buffer_size = total_tokens
            new_buffer = torch.zeros(
                (buffer_size, num_layers, d_in),
                dtype=self.dtype,  # type: ignore
                device=self.device,
            )
            n_tokens_filled = 0
            
            while n_tokens_filled < buffer_size:
                if not os.path.exists(f"{self.cached_activations_path}/{self.next_cache_idx}.safetensors"):
                    print("\n\nWarning: Ran out of cached activation files earlier than expected.")
                    print(f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}.")
                    if buffer_size % self.total_training_tokens != 0:
                        print("This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens")
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    new_buffer = new_buffer[:n_tokens_filled, ...]
                    return new_buffer, torch.zeros(0, num_layers, d_in, dtype=self.dtype, device=self.device)

                activations = self.load_buffer(f"{self.cached_activations_path}/{self.next_cache_idx}.safetensors")
                taking_subset_of_file = False
                if n_tokens_filled + activations.shape[0] > buffer_size:
                    activations = activations[: buffer_size - n_tokens_filled, ...]
                    taking_subset_of_file = True

                new_buffer[n_tokens_filled : n_tokens_filled + activations.shape[0], ...] = activations

                if taking_subset_of_file:
                    self.next_idx_within_buffer = activations.shape[0]
                else:
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0

                n_tokens_filled += activations.shape[0]
            return new_buffer, torch.zeros(0, num_layers, d_in, dtype=self.dtype, device=self.device)

        refill_iterator = range(0, total_tokens, batch_size * context_size)
        new_control_buffer = torch.zeros(
            (total_tokens, num_layers, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )
        new_main_buffer = torch.zeros(
            (total_tokens, num_layers, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )

        control_activations_list = []
        main_activations_list = []

        for refill_batch_idx_start in refill_iterator:
            if self.control_dataset:
                main_batch_size = int(batch_size * (1 - self.control_mixture))
                control_batch_size = batch_size - main_batch_size

                try:
                    # Get tokens from main and control datasets
                    refill_main_batch_tokens = self.get_batch_tokens(main_batch_size).to(self.model.cfg.device)
                    refill_control_batch_tokens = self.get_control_batch_tokens(control_batch_size).to(self.model.cfg.device)
                except StopIteration:
                    print("Warning: Dataset exhausted during refill. Exiting refill loop.")
                    break

                if refill_main_batch_tokens.shape[0] == 0 or refill_control_batch_tokens.shape[0] == 0:
                    print("Warning: Empty batch tokens encountered. Exiting refill loop.")
                    break

                # Combine the tokens from both datasets into a single batch
                combined_batch_tokens = torch.cat((refill_control_batch_tokens, refill_main_batch_tokens), dim=0)

                # Get the activations for the combined tokens in a single forward pass
                combined_activations = self.get_activations(combined_batch_tokens) # [batch seq layer d_in]

                # Split the combined activations back into control and main activations
                control_activations = rearrange(combined_activations[:control_batch_size],
                                                "batch seq layer d_in -> (batch seq) layer d_in")
                main_activations = rearrange(combined_activations[control_batch_size:],
                                            "batch seq layer d_in -> (batch seq) layer d_in")
                
                # Append the activations to their respective lists
                control_activations_list.append(control_activations)
                main_activations_list.append(main_activations)

            else:
                try:
                    # Get tokens from the main dataset
                    refill_batch_tokens = self.get_batch_tokens(batch_size).to(self.model.cfg.device)
                except StopIteration:
                    print("Warning: Dataset exhausted during refill. Exiting refill loop.")
                    break

                if refill_batch_tokens.shape[0] == 0:
                    print("Warning: Empty batch tokens encountered. Exiting refill loop.")
                    break

                # Get the activations for the tokens
                refill_batch_activations = self.get_activations(refill_batch_tokens).view(-1, num_layers, d_in)
                main_activations_list.append(refill_batch_activations)

        # Concatenate the control and main activations
        if self.control_dataset and len(control_activations_list) > 0:
            control_activations = torch.cat(control_activations_list, dim=0)
            control_activations = control_activations[torch.randperm(control_activations.shape[0])]
            new_control_buffer[:control_activations.shape[0]] = control_activations

        if len(main_activations_list) > 0:
            main_activations = torch.cat(main_activations_list, dim=0)
            main_activations = main_activations[torch.randperm(main_activations.shape[0])]
            new_main_buffer[:main_activations.shape[0]] = main_activations

        return new_control_buffer, new_main_buffer


    def get_data_loader(self) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).
        """

        batch_size = self.train_batch_size_tokens

        # Get the new control and main buffers
        new_control_buffer, new_main_buffer = self.get_buffer(self.n_batches_in_buffer // 2)

        control_size = int(self.control_mixture * batch_size)
        main_size = batch_size - control_size

        def mix_activations():
            """
            Generator to yield mixed activations batches.
            Each batch will have `control_size` elements from the control dataset
            and `main_size` elements from the main dataset.
            """
            control_idx = 0
            main_idx = 0

            batch_count = 0
            while control_idx < new_control_buffer.shape[0] and main_idx < new_main_buffer.shape[0]:
                control_batch = new_control_buffer[control_idx:control_idx + control_size]
                main_batch = new_main_buffer[main_idx:main_idx + main_size]
                if control_batch.shape[0] < control_size or main_batch.shape[0] < main_size:
                    break  # Exit if we have incomplete batches

                control_indices = torch.randperm(control_batch.shape[0])
                main_indices = torch.randperm(main_batch.shape[0])

                mixed_batch = torch.cat((control_batch[control_indices], main_batch[main_indices]), dim=0)
                control_idx += control_size
                main_idx += main_size

                batch_count += 1
                yield mixed_batch

        # Create the mixing buffer by combining the control and main buffers
        mixing_buffer = torch.cat([new_control_buffer, new_main_buffer], dim=0)

        # 2. Put 50% in storage
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. Use the other 50% to create a dataloader
        mixed_activations = list(mix_activations())

        if len(mixed_activations) == 0:
            print("Warning: No mixed activations were generated.")
            return None


        # Wrap the mixed activations in a TensorDataset
        dataset = TensorDataset(torch.stack(mixed_activations))

        # Create DataLoader with batch_size=1 to yield individual mixed batches
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0]  # Extract the tensor from the tuple
        )
 
        return iter(dataloader)


    def save_buffer(self, buffer: torch.Tensor, path: str):
        """
        Used by cached activations runner to save a buffer to disk.
        For reuse by later workflows.
        """
        save_file({"activations": buffer}, path)

    def load_buffer(self, path: str) -> torch.Tensor:

        with safe_open(path, framework="pt", device=str(self.device)) as f:  # type: ignore
            buffer = f.get_tensor("activations")
        return buffer

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            batch =  next(self.dataloader)[0]
            return batch
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            return next(self.dataloader)[0] # index [0] because next(dataloader) is for some reason a tuple containing the tensor we want

    def state_dict(self) -> dict[str, torch.Tensor]:
        result = {
            "n_dataset_processed": torch.tensor(self.n_dataset_processed),
        }
        if self._storage_buffer is not None:  # first time might be None
            result["storage_buffer"] = self._storage_buffer
        return result

    def save(self, file_path: str):
        save_file(self.state_dict(), file_path)

    def _get_next_dataset_tokens(self) -> torch.Tensor:
        device = self.device
        if not self.is_dataset_tokenized:
            while True:
                try:
                    s = next(self.iterable_dataset)[self.tokens_column]
                    if s is not None:
                        break
                except StopIteration:
                    # Handle the case where there are no more elements in the iterator
                    s = ""
                    break

            # if s is None:
            #     s = ""
            # assert not (s is None), "Clean your dataset"
            tokens = (
                self.model.to_tokens(
                    s,
                    truncate=False,
                    move_to_device=True,
                    prepend_bos=self.prepend_bos,
                )
                .squeeze(0)
                .to(device)
            )
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
        else:
            tokens = torch.tensor(
                next(self.iterable_dataset)[self.tokens_column],
                dtype=torch.long,
                device=device,
                requires_grad=False,
            )
            if (
                not self.prepend_bos
                and tokens[0] == self.model.tokenizer.bos_token_id  # type: ignore
            ):
                tokens = tokens[1:]
        self.n_dataset_processed += 1
        return tokens

    def _get_next_control_dataset_tokens(self) -> torch.Tensor:
        device = self.device
        if not self.is_control_dataset_tokenized:
            s = next(self.iterable_control_dataset)[self.control_tokens_column]
            tokens = (
                self.model.to_tokens(
                    s,
                    truncate=False,
                    move_to_device=True,
                    prepend_bos=self.prepend_bos,
                )
                .squeeze(0)
                .to(device)
            )
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
        else:
            tokens = torch.tensor(
                next(self.iterable_control_dataset)[self.control_tokens_column],
                dtype=torch.long,
                device=device,
                requires_grad=False,
            )
            if self.prepend_bos:
                bos_token_id_tensor = torch.tensor(
                    [self.model.tokenizer.bos_token_id],
                    device=tokens.device,
                    dtype=torch.long,
                )
                if tokens[0] != bos_token_id_tensor:
                    tokens = torch.cat(
                        (
                            bos_token_id_tensor,
                            tokens,
                        ),
                        dim=0,
                    )
        self.n_dataset_processed += 1
        return tokens

