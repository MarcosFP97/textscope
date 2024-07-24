import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from .config import PROFILES

class RelevanceAnalyzer:
    def __init__(
        self
    )-> None:
        self.profiles = PROFILES
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.task = 'Given a query, determine whether the text is relevant for the query'
    
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
        text:str,
        profile:str
    )-> float:
        if not text:
            return 0.
        
        if profile not in PROFILES:
            raise ValueError(f"Profile '{profile}' not found in the configuration.")
        keywords = PROFILES[profile]
        concat_q = ' '.join(keywords)

        instruct = [
            self.__get_detailed_instruct(self.task, concat_q),
        ]
        doc = [
            text
        ]
        input = instruct + doc
        batch_dict = self.tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.__average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        return scores.tolist()[0][0]
    
    def filter_corpus(
        self    
    ):
       pass ## TO-DO 
   