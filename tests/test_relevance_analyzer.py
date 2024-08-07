import pytest
from textscope.relevance_analyzer import RelevanceAnalyzer

sample_text = 'La adicción al juego es una enfermedad, pero es la única enfermedad que te puede hacer rico. La artritis no te va a hacer ganar un centavo'

@pytest.fixture
def analyzer():
    return RelevanceAnalyzer()

def test_analyze_relevance(analyzer):
    score = analyzer.analyze(sample_text, "gambling")
    print(f"Relevance score for sample text: {score}")
    assert isinstance(score, float), "The result should be a float"
    assert 0 <= score <= 100, "The result should be between 0 and 100"

def test_empty_text(analyzer):
    score = analyzer.analyze("", "gambling")
    assert isinstance(score, float), "The result should be a float"
    assert score == 0, "The result should be 0 for an empty text"