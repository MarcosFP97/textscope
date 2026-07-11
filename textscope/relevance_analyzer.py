from typing import Optional, Union

import torch
from torch import Tensor

from .config import PROFILES
from .e5_backend import (
    DEFAULT_E5_MODEL_NAME,
    DEFAULT_E5_MODEL_REVISION,
    E5Backend,
    get_e5_backend,
)


class RelevanceAnalyzer:
    def __init__(
        self,
        backend: Optional[E5Backend] = None,
        model_name: str = DEFAULT_E5_MODEL_NAME,
        revision: str = DEFAULT_E5_MODEL_REVISION,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.profiles = PROFILES
        self.backend = backend or get_e5_backend(
            model_name=model_name,
            revision=revision,
            device=device,
        )
        # Keep the original public attributes for callers that inspect them.
        self.tokenizer = self.backend.tokenizer
        self.model = self.backend.model
        self.device = self.backend.device
        self.task = 'Given a query, determine whether the text is relevant for the query'
        # Cache for profile query embeddings
        self._profile_cache = {}

    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        return self.backend._average_pool(last_hidden_states, attention_mask)

    def _get_profile_embedding(self, profile: str) -> Tensor:
        """Get or compute cached embedding for a profile query."""
        if profile in self._profile_cache:
            return self._profile_cache[profile]

        keywords = self.profiles[profile]
        concat_q = ' '.join(keywords)
        instruct_text = self._get_detailed_instruct(self.task, concat_q)

        embedding = self.backend.embed([instruct_text])

        self._profile_cache[profile] = embedding
        return embedding

    def analyze(self, text: str, profile: str) -> float:
        if not text:
            return 0.

        if profile not in self.profiles:
            raise ValueError(f"Profile '{profile}' not found in the configuration.")

        profile_emb = self._get_profile_embedding(profile)

        text_emb = self.backend.embed([text])

        scores = (profile_emb @ text_emb.T) * 100
        return scores.tolist()[0][0]
