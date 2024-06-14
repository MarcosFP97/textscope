import pytest
from textscope.relevance_analyzer import RelevanceAnalyzer

model_name = 'intfloat/multilingual-e5-large-instruct'
sample_text = 'La adicción al juego es una enfermedad, pero es la única enfermedad que te puede hacer rico. La artritis no te va a hacer ganar un centavo'

@pytest.fixture
def analyzer():
    return RelevanceAnalyzer(model_name)

def test_analyze_relevance(analyzer):
    score = analyzer.analyze(sample_text)
    print(f"Relevance score for sample text: {score}")
    assert isinstance(score, float), "El resultado debería ser un float"
    assert 0 <= score <= 100, "El score debería estar entre 0 y 100"

def test_empty_text(analyzer):
    score = analyzer.analyze("")
    assert isinstance(score, float), "El resultado debería ser un float"
    assert score == 0, "El score debería ser 0 para un texto vacío"