import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from .config import SUBTHEMES

class SubthemeAnalyzer:
    def __init__(
            self, 
            model_name: str,
            task: str = 'Given a topic, determine whether the text discusses the topic'
    )-> None:
        self.subthemes = SUBTHEMES
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.task = task

    def __get_detailed_instruct(
        self,
        task_description: str, 
        query: str
    ) -> str:
        return f'Instruct: {task_description}\Topic: {query}'
    
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
    )-> dict:
        if not text:
            return {}

        sentences = sent_tokenize(text)
        subtheme_scores = {}

        for k,v in self.subthemes.items():
            max_sim = 0.
            for sent in sentences:
                if self.task:
                    instruct = [
                        self.__get_detailed_instruct(self.task, v),
                    ]
                    doc = [
                        sent
                    ]
                    input = instruct + doc
                else:
                    input = f'Query: {v} Passage: {sent}'

                batch_dict = self.tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                outputs = self.model(**batch_dict)
                embeddings = self.__average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                scores = (embeddings[:1] @ embeddings[1:].T) * 100
                sim = scores.tolist()[0][0]

                if sim > max_sim:
                    max_sim = sim

            subtheme_scores[k] = max_sim

        return subtheme_scores
    
    def analyze_bin(
            self, 
            text:str,
            thr:float=86.
    )-> dict:
        if not text:
            return {}
        
        sentences = sent_tokenize(text)
        subtheme_pres = {}

        for k,v in self.subthemes.items():
            max_sim = 0.
            for sent in sentences:
                if self.task:
                    instruct = [
                        self.__get_detailed_instruct(self.task, v),
                    ]
                    doc = [
                        sent
                    ]
                    input = instruct + doc
                else:
                    input = f'Query: {v} Passage: {sent}'

                batch_dict = self.tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                outputs = self.model(**batch_dict)
                embeddings = self.__average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                scores = (embeddings[:1] @ embeddings[1:].T) * 100
                sim = scores.tolist()[0][0]

                if sim > max_sim:
                    max_sim = sim

            if max_sim > thr:
                subtheme_pres[k] = 1
            else:
                subtheme_pres[k] = 0

        return subtheme_pres

