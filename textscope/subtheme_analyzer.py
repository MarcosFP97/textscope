from sentence_transformers import SentenceTransformer
import torch
from nltk.tokenize import sent_tokenize
from .config import SUBTHEMES

class SubthemeAnalyzer:
    def __init__(
            self, 
            sbert_model:str
    )-> None:
        self.subthemes = SUBTHEMES
        self.model = SentenceTransformer(sbert_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def analyze(
            self, 
            text:str
    )-> dict:
        if not text:
            return {}
        
        sentences = sent_tokenize(text)
        subtheme_scores = {}

        for k,v in self.subthemes.items():
            query = self.model.encode(v)
            max_sim = 0.
            
            for sent in sentences:
                s = self.model.encode(sent)
                sim = self.model.similarity(query,s)
                sim = sim.item()

                if sim > max_sim:
                    max_sim = sim

            subtheme_scores[k] = max_sim

        return subtheme_scores
    
    def analyze_bin(
            self, 
            text:str,
            thr:float=0.3
    )-> dict:
        if not text:
            return {}
        
        sentences = sent_tokenize(text)
        subtheme_pres = {}

        for k,v in self.subthemes.items():
            query = self.model.encode(v)
            max_sim = 0.
            
            for sent in sentences:
                s = self.model.encode(sent)
                sim = self.model.similarity(query,s)
                sim = sim.item()

                if sim > max_sim:
                    max_sim = sim

            if max_sim > thr:
                subtheme_pres[k] = 1
            else:
                subtheme_pres[k] = 0

        return subtheme_pres

