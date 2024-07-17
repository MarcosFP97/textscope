import pytest
from textscope.subtheme_analyzer import SubthemeAnalyzer

model_name = 'hiiamsid/sentence_similarity_spanish_es'
sample_text = 'La adicción al juego es una enfermedad, pero es la única enfermedad que te puede hacer rico. La artritis no te va a hacer ganar un centavo'

@pytest.fixture
def analyzer():
    return SubthemeAnalyzer(model_name)

