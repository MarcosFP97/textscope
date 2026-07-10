import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from .config import SUBTHEMES
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class SubthemeAnalyzer:
    def __init__(self) -> None:
        self.subthemes = SUBTHEMES
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.task = 'Given a set of words forming a topic, determine whether the text discusses the topic'
        # Cache for keyword embeddings and index mappings per profile
        self._kw_cache = {}

    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nTopic: {query}'

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed_batch(self, texts: list, batch_size: int = 32) -> Tensor:
        """Embed a list of texts in batches, returning normalized embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_dict = self.tokenizer(
                batch, max_length=512, padding=True, truncation=True, return_tensors='pt',
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            embeddings = self._average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def _get_kw_cache(self, profile: str):
        """Get or compute cached keyword embeddings and index mapping for a profile."""
        if profile in self._kw_cache:
            return self._kw_cache[profile]

        subthemes = SUBTHEMES[profile]
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

        kw_embeddings = self._embed_batch(keyword_texts)
        self._kw_cache[profile] = (kw_embeddings, subtheme_indices)
        return kw_embeddings, subtheme_indices

    def _analyze_common(self, text: str, profile: str):
        """Shared logic for analyze and analyze_bin."""
        if profile not in SUBTHEMES:
            raise ValueError(f"Profile '{profile}' not found in the subthemes configuration.")
        n_subthemes = len(SUBTHEMES[profile])

        sentences = sent_tokenize(text)
        if not sentences:
            return None, n_subthemes

        sent_embeddings = self._embed_batch(sentences)
        kw_embeddings, subtheme_indices = self._get_kw_cache(profile)
        sim_matrix = (kw_embeddings @ sent_embeddings.T) * 100

        max_sims = []
        for idx in range(n_subthemes):
            kw_idx = subtheme_indices[idx]
            max_sims.append(sim_matrix[kw_idx].max().item())

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

    def filter_corpus(self):
        pass  ## TO-DO
