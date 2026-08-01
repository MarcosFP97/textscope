"""Shared E5 model loading and embedding primitives."""

from functools import lru_cache
from typing import Iterator, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


DEFAULT_E5_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
DEFAULT_E5_MODEL_REVISION = "274baa43b0e13e37fafa6428dbc7938e62e5c439"
DEFAULT_E5_BATCH_SIZE = 32


class E5Backend:
    """Own one tokenizer/model pair and expose normalized E5 embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_E5_MODEL_NAME,
        revision: str = DEFAULT_E5_MODEL_REVISION,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not revision:
            raise ValueError("revision must not be empty")

        self.model_name = model_name
        self.revision = revision
        self.device = _resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=False,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_components(
        cls,
        tokenizer,
        model,
        device: Union[str, torch.device],
        model_name: str = "preloaded",
        revision: str = "preloaded",
    ) -> "E5Backend":
        """Wrap already-loaded components for backwards-compatible subclasses."""
        backend = cls.__new__(cls)
        backend.model_name = model_name
        backend.revision = revision
        backend.device = _resolve_device(device)
        backend.tokenizer = tokenizer
        backend.model = model
        backend.model.eval()
        return backend

    @staticmethod
    def _average_pool(
        last_hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )
        token_counts = attention_mask.sum(dim=1)[..., None].clamp_min(1)
        return last_hidden.sum(dim=1) / token_counts

    def iter_embeddings(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_E5_BATCH_SIZE,
    ) -> Iterator[Tensor]:
        """Yield normalized embeddings one device-resident batch at a time."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_dict = self.tokenizer(
                batch,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_dict = {
                key: value.to(self.device)
                for key, value in batch_dict.items()
            }
            with torch.inference_mode():
                outputs = self.model(**batch_dict)
                embeddings = self._average_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                )
                yield F.normalize(embeddings, p=2, dim=1)

    def embed(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_E5_BATCH_SIZE,
    ) -> Tensor:
        """Embed a finite non-empty collection and concatenate its batches."""
        batches = list(self.iter_embeddings(texts, batch_size=batch_size))
        if not batches:
            raise ValueError("texts must contain at least one item")
        if len(batches) == 1:
            return batches[0]
        return torch.cat(batches, dim=0)


def _resolve_device(
    device: Optional[Union[str, torch.device]],
) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


@lru_cache(maxsize=None)
def _get_cached_e5_backend(
    model_name: str,
    revision: str,
    device_name: str,
) -> E5Backend:
    return E5Backend(
        model_name=model_name,
        revision=revision,
        device=device_name,
    )


def get_e5_backend(
    model_name: str = DEFAULT_E5_MODEL_NAME,
    revision: str = DEFAULT_E5_MODEL_REVISION,
    device: Optional[Union[str, torch.device]] = None,
) -> E5Backend:
    """Return the process-wide backend for a model/revision/device tuple."""
    resolved_device = _resolve_device(device)
    return _get_cached_e5_backend(
        model_name,
        revision,
        str(resolved_device),
    )
