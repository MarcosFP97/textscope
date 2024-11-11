import torch
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from .config import SUBTHEMES
import nltk
nltk.download('punkt_tab')

class SubthemeAnalyzer:
    def __init__(
        self
    )-> None:
        self.subthemes = SUBTHEMES
        model_name = 'intfloat/multilingual-e5-small'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # self.task = 'Given a topic, determine whether the text discusses the topic'

    # def __get_detailed_instruct(
    #     self,
    #     task_description: str, 
    #     query: str
    # ) -> str:
    #     return f'Instruct: {task_description}\Topic: {query}'
    
    def __average_pool(
        self,
        last_hidden_states: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __main_analysis(
        self,
        theme:str,
        sent:str
    )-> float:
        # instruct = [
        #     self.__get_detailed_instruct(self.task, theme),
        # ]
        # doc = [
        #     sent
        # ]
        theme = 'query: '+theme
        sent = 'passage: '+sent
        input_texts = [theme, sent]
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        outputs = self.model(**batch_dict)
        embeddings = self.__average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        sim = scores.tolist()[0][0]
        return sim

    def analyze(
        self, 
        text:str,
        profile:str
    )-> list:
        if not text:
            return []

        if profile not in SUBTHEMES:
            raise ValueError(f"Profile '{profile}' not found in the subthemes configuration.")
        subthemes = SUBTHEMES[profile]

        sentences = sent_tokenize(text)
        subthemes_scores = []
        logs = []
        for theme in subthemes:
            max_sim = 0.
            max_sent = ""
            for sent in sentences:
                sim = self.__main_analysis(theme, sent)
                if sim > max_sim:
                    max_sim = sim
                    max_sent = sent
            log = f'Theme: {theme} Sentence: {max_sent} Sim: {max_sim}'
            logs.append(log)
            subthemes_scores.append(max_sim)

        return subthemes_scores
    
    def analyze_bin(
        self, 
        text:str,
        profile:str,
        thr:float=86.
    )-> dict:
        if not text:
            return []

        if profile not in SUBTHEMES:
            raise ValueError(f"Profile '{profile}' not found in the subthemes configuration.")
        subthemes = SUBTHEMES[profile]
        
        sentences = sent_tokenize(text)
        subtheme_pres = []
        for theme in subthemes:
            max_sim = 0.
            for sent in sentences:
                sim = self.__main_analysis(theme, sent)
                if sim > max_sim:
                    max_sim = sim

            if max_sim > thr:
                subtheme_pres.append(1)
            else:
                subtheme_pres.append(0)

        return subtheme_pres

