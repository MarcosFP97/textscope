import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from .config import GAMBLING_THEMES

class RelevanceAnalyzer:
    def __init__(
            self, 
            model_name: str
    )-> None:
        self.themes = GAMBLING_THEMES
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.task = 'Given a search query, determine whether the text is relevant for the query'
    
    def __get_detailed_instruct(
        self,
        task_description: str, 
        query: str
    ) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def __average_pool(
            self,
            last_hidden_states: Tensor,
            attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def analyze(
            self, 
            text:str
    )-> float:
        if not text:
            return 0.
        
        # Lógica para analizar la relevancia del texto con respecto a una serie de queries que modelan la temática
        rels = []
        for k in self.themes:
            instruct = [
                self.__get_detailed_instruct(self.task, k),
            ]
            documents = [
                text
            ]
            input = instruct + documents
            batch_dict = self.tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = self.__average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:1] @ embeddings[1:].T) * 100
            rels.append(scores.tolist()[0][0])

        return np.mean(rels)