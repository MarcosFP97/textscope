from typing import Optional, Union

import torch
from torch import Tensor
from nltk.tokenize import sent_tokenize

from .config import SUBTHEMES
from .e5_backend import (
    DEFAULT_E5_BATCH_SIZE,
    DEFAULT_E5_MODEL_NAME,
    DEFAULT_E5_MODEL_REVISION,
    E5Backend,
    get_e5_backend,
)


class SubthemeAnalyzer:
    def __init__(
        self,
        backend: Optional[E5Backend] = None,
        model_name: str = DEFAULT_E5_MODEL_NAME,
        revision: str = DEFAULT_E5_MODEL_REVISION,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = DEFAULT_E5_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.subthemes = SUBTHEMES
        self.backend = backend or get_e5_backend(
            model_name=model_name,
            revision=revision,
            device=device,
        )
        # Keep the original public attributes for callers that inspect them.
        self.tokenizer = self.backend.tokenizer
        self.model = self.backend.model
        self.device = self.backend.device
        self.batch_size = batch_size
        self.task = 'Given a set of words forming a topic, determine whether the text discusses the topic'
        # Cache for keyword embeddings and index mappings per profile
        self._kw_cache = {}

    def _get_backend(self) -> E5Backend:
        backend = getattr(self, "backend", None)
        if backend is None:
            # Compatibility with subclasses written against pre-backend versions.
            backend = E5Backend.from_components(
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.device,
                model_name=getattr(self.model, "name_or_path", "preloaded"),
            )
            self.backend = backend
        return backend

    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return self._get_backend()._average_pool(last_hidden_states, attention_mask)

    def _embed_batch(self, texts: list, batch_size: int = DEFAULT_E5_BATCH_SIZE) -> Tensor:
        """Embed a list of texts in batches, returning normalized embeddings."""
        return self._get_backend().embed(texts, batch_size=batch_size)

    def _get_kw_cache(self, profile: str):
        """Get or compute cached keyword embeddings and index mapping for a profile."""
        if profile in self._kw_cache:
            return self._kw_cache[profile]

        subthemes = self.subthemes[profile]
        keyword_texts = []
        subtheme_indices = {}
        is_nested = any(isinstance(item, list) for item in subthemes)

        if is_nested:
            for idx, theme in enumerate(subthemes):
                start = len(keyword_texts)
                for kw in theme:
                    keyword_texts.append(self._get_detailed_instruct(self.task, kw))
                subtheme_indices[idx] = list(range(start, len(keyword_texts)))
        else:
            for idx, theme in enumerate(subthemes):
                keyword_texts.append(self._get_detailed_instruct(self.task, theme))
                subtheme_indices[idx] = [len(keyword_texts) - 1]

        kw_embeddings = self._embed_batch(
            keyword_texts,
            batch_size=getattr(self, "batch_size", DEFAULT_E5_BATCH_SIZE),
        )
        self._kw_cache[profile] = (kw_embeddings, subtheme_indices)
        return kw_embeddings, subtheme_indices

    def _analyze_common(self, text: str, profile: str):
        """Shared logic for analyze and analyze_bin."""
        if profile not in self.subthemes:
            raise ValueError(f"Profile '{profile}' not found in the subthemes configuration.")
        n_subthemes = len(self.subthemes[profile])

        try:
            sentences = sent_tokenize(text, language='spanish')
        except LookupError as exc:
            raise RuntimeError(
                "NLTK sentence tokenizer data is missing. Install it once with "
                "`python -m nltk.downloader punkt_tab`."
            ) from exc
        if not sentences:
            return None, n_subthemes

        kw_embeddings, subtheme_indices = self._get_kw_cache(profile)
        max_sims = [float("-inf")] * n_subthemes
        backend = self._get_backend()
        for sent_embeddings in backend.iter_embeddings(
            sentences,
            batch_size=getattr(self, "batch_size", DEFAULT_E5_BATCH_SIZE),
        ):
            sim_matrix = (kw_embeddings @ sent_embeddings.T) * 100
            for idx in range(n_subthemes):
                kw_idx = subtheme_indices[idx]
                batch_max = sim_matrix[kw_idx].max().item()
                max_sims[idx] = max(max_sims[idx], batch_max)

        return max_sims, n_subthemes

    def analyze(self, text: str, profile: str) -> list:
        if not text:
            return []
        result = self._analyze_common(text, profile)
        if result[0] is None:
            return [0.0] * result[1]
        return result[0]

    def analyze_bin(self, text: str, profile: str, thr: float = 85.) -> list:
        if not text:
            return []
        result = self._analyze_common(text, profile)
        if result[0] is None:
            return [0] * result[1]
        return [1 if s > thr else 0 for s in result[0]]
