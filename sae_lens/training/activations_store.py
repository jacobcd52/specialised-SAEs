from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Generator, Iterator, Literal, cast

import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import HfHubHTTPError
from requests import HTTPError
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from einops import rearrange

from sae_lens.config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.sae import SAE
from sae_lens.tokenization_and_batching import concat_and_batch_sequences


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
    control_tokens_column: Literal["tokens", "input_ids", "text", "problem"]
    tokens_column: Literal["tokens", "input_ids", "text", "problem"]
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
        override_dataset: HfDataset | None = None,
    ) -> "ActivationsStore":
        cached_activations_path = cfg.cached_activations_path
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None

        if override_dataset is None and cfg.dataset_path == "":
            raise ValueError(
                "You must either pass in a dataset or specify a dataset_path in your configutation."
            )

        return cls(
            model=model,
            dataset=override_dataset or cfg.dataset_path,
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

    @classmethod
    def from_sae(
        cls,
        model: HookedRootModule,
        sae: SAE,
        context_size: int | None = None,
        dataset: HfDataset | str | None = None,
        streaming: bool = True,
        store_batch_size_prompts: int = 8,
        n_batches_in_buffer: int = 8,
        train_batch_size_tokens: int = 4096,
        total_tokens: int = 10**9,
        device: str = "cpu",
    ) -> "ActivationsStore":

        return cls(
            model=model,
            dataset=sae.cfg.dataset_path if dataset is None else dataset,
            d_in=sae.cfg.d_in,
            hook_name=sae.cfg.hook_name,
            hook_layer=sae.cfg.hook_layer,
            hook_head_index=sae.cfg.hook_head_index,
            context_size=sae.cfg.context_size if context_size is None else context_size,
            prepend_bos=sae.cfg.prepend_bos,
            streaming=streaming,
            store_batch_size_prompts=store_batch_size_prompts,
            train_batch_size_tokens=train_batch_size_tokens,
            n_batches_in_buffer=n_batches_in_buffer,
            total_training_tokens=total_tokens,
            normalize_activations=sae.cfg.normalize_activations,
            dataset_trust_remote_code=sae.cfg.dataset_trust_remote_code,
            dtype=sae.cfg.dtype,
            device=torch.device(device),
        )

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
        # if isinstance(dataset, (Dataset, DatasetDict)):
        #     self.dataset = cast(Dataset | DatasetDict, self.dataset)
        # MAYBE NEED THIS
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
        self.half_buffer_size = n_batches_in_buffer // 2
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
        elif "problem" in dataset_sample.keys():
            self.is_dataset_tokenized = False
            self.tokens_column = "problem"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', or 'text' column."
            )
        if self.is_dataset_tokenized:
            ds_context_size = len(dataset_sample[self.tokens_column])
            if ds_context_size != self.context_size:
                raise ValueError(
                    f"pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}."
                )
            # TODO: investigate if this can work for iterable datasets, or if this is even worthwhile as a perf improvement
            if hasattr(self.dataset, "set_format"):
                self.dataset.set_format(type="torch", columns=[self.tokens_column])  # type: ignore

            if (
                isinstance(dataset, str)
                and hasattr(model, "tokenizer")
                and model.tokenizer is not None
            ):
                validate_pretokenized_dataset_tokenizer(
                    dataset_path=dataset, model_tokenizer=model.tokenizer
                )
        else:
            print(
                "Warning: Dataset is not tokenized. Pre-tokenizing will improve performance and allows for more control over special tokens. See https://jbloomaus.github.io/SAELens/training_saes/#pretokenizing-datasets for more info."
            )

        self.iterable_sequences = self._iterate_tokenized_sequences()

        self.check_cached_activations_against_config()




    def _iterate_raw_dataset(
        self,
    ) -> Generator[torch.Tensor | list[int] | str, None, None]:
        """
        Helper to iterate over the dataset while incrementing n_dataset_processed
        """
        for row in self.dataset:
            # typing datasets is difficult
            yield row[self.tokens_column]  # type: ignore
            self.n_dataset_processed += 1

    def _iterate_raw_dataset_tokens(self) -> Generator[torch.Tensor, None, None]:
        """
        Helper to create an iterator which tokenizes raw text from the dataset on the fly
        """
        for row in self._iterate_raw_dataset():
            tokens = (
                self.model.to_tokens(
                    row,
                    truncate=False,
                    move_to_device=True,
                    prepend_bos=False,
                )
                .squeeze(0)
                .to(self.device)
            )
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
            yield tokens

    def _iterate_tokenized_sequences(self) -> Generator[torch.Tensor, None, None]:
        """
        Generator which iterates over full sequence of context_size tokens
        """
        # If the datset is pretokenized, we can just return each row as a tensor, no further processing is needed.
        # We assume that all necessary BOS/EOS/SEP tokens have been added during pretokenization.
        if self.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                yield torch.tensor(
                    row,
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
        # If the dataset isn't tokenized, we'll tokenize, concat, and batch on the fly
        else:
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_token_id = None if tokenizer is None else tokenizer.bos_token_id
            yield from concat_and_batch_sequences(
                tokens_iterator=self._iterate_raw_dataset_tokens(),
                context_size=self.context_size,
                begin_batch_token_id=(bos_token_id if self.prepend_bos else None),
                begin_sequence_token_id=None,
                sequence_separator_token_id=(
                    bos_token_id if self.prepend_bos else None
                ),
            )

    def _iterate_raw_dataset(
        self,
    ) -> Generator[torch.Tensor | list[int] | str, None, None]:
        """
        Helper to iterate over the dataset while incrementing n_dataset_processed
        """
        for row in self.dataset:
            # typing datasets is difficult
            yield row[self.tokens_column]  # type: ignore
            self.n_dataset_processed += 1

    def _iterate_raw_dataset_tokens(self) -> Generator[torch.Tensor, None, None]:
        """
        Helper to create an iterator which tokenizes raw text from the dataset on the fly
        """
        for row in self._iterate_raw_dataset():
            tokens = (
                self.model.to_tokens(
                    row,
                    truncate=False,
                    move_to_device=True,
                    prepend_bos=False,
                )
                .squeeze(0)
                .to(self.device)
            )
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
            yield tokens

    def _iterate_tokenized_sequences(self) -> Generator[torch.Tensor, None, None]:
        """
        Generator which iterates over full sequence of context_size tokens
        """
        # If the datset is pretokenized, we can just return each row as a tensor, no further processing is needed.
        # We assume that all necessary BOS/EOS/SEP tokens have been added during pretokenization.
        if self.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                yield torch.tensor(
                    row,
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
        # If the dataset isn't tokenized, we'll tokenize, concat, and batch on the fly
        else:
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_token_id = None if tokenizer is None else tokenizer.bos_token_id
            yield from concat_and_batch_sequences(
                tokens_iterator=self._iterate_raw_dataset_tokens(),
                context_size=self.context_size,
                begin_batch_token_id=(bos_token_id if self.prepend_bos else None),
                begin_sequence_token_id=None,
                sequence_separator_token_id=(
                    bos_token_id if self.prepend_bos else None
                ),
            )

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

    def shuffle_input_dataset(self, seed: int, buffer_size: int = 1):
        """
        This applies a shuffle to the huggingface dataset that is the input to the activations store. This
        also shuffles the shards of the dataset, which is especially useful for evaluating on different
        sections of very large streaming datasets. Buffer size is only relevant for streaming datasets.
        The default buffer_size of 1 means that only the shard will be shuffled; larger buffer sizes will
        additionally shuffle individual elements within the shard.
        """
        if type(self.dataset) == IterableDataset:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.iterable_dataset = iter(self.dataset)

    def reset_input_dataset(self):
        """
        Resets the input dataset iterator to the beginning.
        """
        self.iterable_dataset = iter(self.dataset)

    @property
    def storage_buffer(self) -> torch.Tensor:
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.half_buffer_size)

        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader

    def get_batch_tokens(
        self, batch_size: int | None = None, raise_at_epoch_end: bool = False
    ):
        """
        Streams a batch of tokens from the main dataset.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        sequences = []
        # the sequences iterator yields fully formed tokens of size context_size, so we just need to cat these into a batch
        for _ in range(batch_size):
            try:
                sequences.append(next(self.iterable_sequences))
            except StopIteration:
                self.iterable_sequences = self._iterate_tokenized_sequences()
                if raise_at_epoch_end:
                    raise StopIteration(
                        f"Ran out of tokens in dataset after {self.n_dataset_processed} samples, beginning the next epoch."
                    )
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
                prepend_bos=False,
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
            try:
                stacked_activations[:, :, 0] = layerwise_activations[
                    self.hook_name
                ].view(n_batches, n_context, -1)
            except RuntimeError as e:
                print(f"Error during view operation: {e}")
                print("Attempting to use reshape instead...")
                stacked_activations[:, :, 0] = layerwise_activations[
                    self.hook_name
                ].reshape(n_batches, n_context, -1)
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

        control_size = int(total_tokens * self.control_mixture) if self.control_dataset and self.control_mixture else 0
        main_size = total_tokens - control_size

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

        new_control_buffer = torch.zeros(
            (control_size, num_layers, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )
        new_main_buffer = torch.zeros(
            (main_size, num_layers, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )

        control_activations_list = []
        main_activations_list = []

        refill_iterator = range(0, total_tokens, batch_size * context_size)

        for refill_batch_idx_start in refill_iterator:
            if self.control_dataset and self.control_mixture:
                main_batch_size = int(batch_size * (1 - self.control_mixture))
                control_batch_size = batch_size - main_batch_size


                try:
                    refill_main_batch_tokens = self.get_batch_tokens(main_batch_size).to(self.model.cfg.device)
                    refill_control_batch_tokens = self.get_control_batch_tokens(control_batch_size).to(self.model.cfg.device)
                except StopIteration:
                    print("Warning: Dataset exhausted during refill. Exiting refill loop.")
                    break

                if refill_main_batch_tokens.shape[0] == 0 or refill_control_batch_tokens.shape[0] == 0:
                    print("Warning: Empty batch tokens encountered. Exiting refill loop.")
                    break

                combined_batch_tokens = torch.cat((refill_control_batch_tokens, refill_main_batch_tokens), dim=0)

                combined_activations = self.get_activations(combined_batch_tokens)  # [batch seq layer d_in]

                control_activations = rearrange(combined_activations[:control_batch_size],
                                                "batch seq layer d_in -> (batch seq) layer d_in")
                main_activations = rearrange(combined_activations[control_batch_size:],
                                            "batch seq layer d_in -> (batch seq) layer d_in")
                
                control_activations_list.append(control_activations)
                main_activations_list.append(main_activations)

            else:
                try:
                    refill_batch_tokens = self.get_batch_tokens(batch_size).to(self.model.cfg.device)
                except StopIteration:
                    print("Warning: Dataset exhausted during refill. Exiting refill loop.")
                    break

                if refill_batch_tokens.shape[0] == 0:
                    print("Warning: Empty batch tokens encountered. Exiting refill loop.")
                    break

                refill_batch_activations = self.get_activations(refill_batch_tokens).view(-1, num_layers, d_in)
                main_activations_list.append(refill_batch_activations)

        if self.control_dataset and self.control_mixture and len(control_activations_list) > 0:
            control_activations = torch.cat(control_activations_list, dim=0)
            control_activations = control_activations[torch.randperm(control_activations.shape[0])]
            new_control_buffer = control_activations

        else:
            new_control_buffer = torch.zeros(0, num_layers, d_in, dtype=self.dtype, device=self.device)

        if len(main_activations_list) > 0:
            main_activations = torch.cat(main_activations_list, dim=0)
            main_activations = main_activations[torch.randperm(main_activations.shape[0])]
            new_main_buffer = main_activations

        else:
            new_main_buffer = torch.zeros(0, num_layers, d_in, dtype=self.dtype, device=self.device)

        return new_control_buffer, new_main_buffer




        # Create the mixing buffer by combining the control and main buffers if control dataset exists
        if self.control_mixture:
            mixing_buffer = torch.cat([new_control_buffer, new_main_buffer], dim=0)
        else:
            mixing_buffer = new_main_buffer

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

    def get_data_loader(self) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).
        """
        batch_size = self.train_batch_size_tokens

        # Get the new control and main buffers
        new_control_buffer, new_main_buffer = self.get_buffer(self.n_batches_in_buffer // 2)

        control_size = int(self.control_mixture * batch_size) if self.control_mixture else 0
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
            while main_idx < new_main_buffer.shape[0]:
                if self.control_mixture and control_idx < new_control_buffer.shape[0]:
                    control_batch = new_control_buffer[control_idx:control_idx + control_size]
                    main_batch = new_main_buffer[main_idx:main_idx + main_size]
                    if control_batch.shape[0] < control_size or main_batch.shape[0] < main_size:
                        break  # Exit if we have incomplete batches

                    control_indices = torch.randperm(control_batch.shape[0])
                    main_indices = torch.randperm(main_batch.shape[0])

                    mixed_batch = torch.cat((control_batch[control_indices], main_batch[main_indices]), dim=0)
                    control_idx += control_size
                else:
                    main_batch = new_main_buffer[main_idx:main_idx + batch_size]
                    if main_batch.shape[0] < batch_size:
                        break  # Exit if we have incomplete batches

                    main_indices = torch.randperm(main_batch.shape[0])
                    mixed_batch = main_batch[main_indices]
                
                main_idx += main_size if self.control_mixture and control_idx < new_control_buffer.shape[0] else batch_size

                batch_count += 1
                yield mixed_batch

        # Create the mixing buffer by combining the control and main buffers if control dataset exists
        if self.control_mixture:
            mixing_buffer = torch.cat([new_control_buffer, new_main_buffer], dim=0)
        else:
            mixing_buffer = new_main_buffer

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


    def validate_pretokenized_dataset_tokenizer(
        dataset_path: str, model_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """
        Helper to validate that the tokenizer used to pretokenize the dataset matches the model tokenizer.
        """
        try:
            tokenization_cfg_path = hf_hub_download(
                dataset_path, "sae_lens.json", repo_type="dataset"
            )
        except HfHubHTTPError:
            return
        if tokenization_cfg_path is None:
            return
        with open(tokenization_cfg_path, "r") as f:
            tokenization_cfg = json.load(f)
        tokenizer_name = tokenization_cfg["tokenizer_name"]
        try:
            ds_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # if we can't download the specified tokenizer to verify, just continue
        except HTTPError:
            return
        if ds_tokenizer.get_vocab() != model_tokenizer.get_vocab():
            raise ValueError(
                f"Dataset tokenizer {tokenizer_name} does not match model tokenizer {model_tokenizer}."
            )
    
    def _get_next_dataset_tokens(self) -> torch.Tensor:
        device = self.device
        if not self.is_dataset_tokenized:
            while True:
                try:
                    s = next(self.iterable_dataset)[self.tokens_column]
                    if s is not None:
                        break
                except StopIteration:
                    print("Reached end of dataset. Resetting iterator.")
                    self.iterable_dataset = iter(self.dataset)
                    s = next(self.iterable_dataset)[self.tokens_column]
                    break

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
            while True:
                try:
                    tokens = torch.tensor(
                        next(self.iterable_dataset)[self.tokens_column],
                        dtype=torch.long,
                        device=device,
                        requires_grad=False,
                    )
                    break
                except StopIteration:
                    print("Reached end of dataset. Resetting iterator.")
                    self.iterable_dataset = iter(self.dataset)

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
            while True:
                try:
                    s = next(self.iterable_control_dataset)[self.control_tokens_column]
                    break
                except StopIteration:
                    print("Reached end of control dataset. Resetting iterator.")
                    self.iterable_control_dataset = iter(self.control_dataset)
                    s = next(self.iterable_control_dataset)[self.control_tokens_column]
                    break

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
            while True:
                try:
                    tokens = torch.tensor(
                        next(self.iterable_control_dataset)[self.control_tokens_column],
                        dtype=torch.long,
                        device=device,
                        requires_grad=False,
                    )
                    break
                except StopIteration:
                    print("Reached end of control dataset. Resetting iterator.")
                    self.iterable_control_dataset = iter(self.control_dataset)

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

