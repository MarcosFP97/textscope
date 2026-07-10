import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
from .config import PROFILES


class RelevanceAnalyzer:
    def __init__(self) -> None:
        self.profiles = PROFILES
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.task = 'Given a query, determine whether the text is relevant for the query'
        # Cache for profile query embeddings
        self._profile_cache = {}

    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_profile_embedding(self, profile: str) -> Tensor:
        """Get or compute cached embedding for a profile query."""
        if profile in self._profile_cache:
            return self._profile_cache[profile]

        keywords = PROFILES[profile]
        concat_q = ' '.join(keywords)
        instruct_text = self._get_detailed_instruct(self.task, concat_q)

        batch_dict = self.tokenizer(
            [instruct_text], max_length=512, padding=True, truncation=True, return_tensors='pt',
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embedding = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)

        self._profile_cache[profile] = embedding
        return embedding

    def analyze(self, text: str, profile: str) -> float:
        if not text:
            return 0.

        if profile not in PROFILES:
            raise ValueError(f"Profile '{profile}' not found in the configuration.")

        profile_emb = self._get_profile_embedding(profile)

        batch_dict = self.tokenizer(
            [text], max_length=512, padding=True, truncation=True, return_tensors='pt',
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        text_emb = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        text_emb = F.normalize(text_emb, p=2, dim=1)

        scores = (profile_emb @ text_emb.T) * 100
        return scores.tolist()[0][0]

    def filter_corpus(self):
        pass  ## TO-DO
