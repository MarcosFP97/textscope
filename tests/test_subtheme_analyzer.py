import pytest
from textscope.subtheme_analyzer import SubthemeAnalyzer

sample_text = 'A prevención da violencia de xénero require detectar situacións de control, maltrato psicolóxico e agresións dentro da parella. A protección das vítimas e a educación en igualdade son fundamentais para combater a violencia machista.'
sample_text_rel_small = 'El acceso a internet debe ser reconocido como un derecho humano básico en la era digital.'
sample_text_rel_large = 'Los derechos digitales son fundamentales para garantizar que las personas puedan expresarse libremente y acceder a información en línea sin temor a censura. Además, es crucial proteger la privacidad de los datos personales frente a corporaciones y gobiernos, asegurando así la seguridad en la navegación y la comunicación digital.'

@pytest.fixture
def analyzer():
    return SubthemeAnalyzer()

def test_analyze_sub(analyzer):
    puncts = analyzer.analyze(sample_text, "violencia_genero_gal")
    print(f"Subthemes scores: {puncts}")
    assert isinstance(puncts, list), "Results should be a list of scores"

def test_analyze_sub_dr(analyzer):
    puncts = analyzer.analyze(sample_text_rel_large, "cibersec")
    print(f"Subthemes scores: {puncts}")
    assert isinstance(puncts, list), "Results should be a list of scores"

def test_analyze_subpres(analyzer):
    puncts = analyzer.analyze_bin(sample_text_rel_large, "cibersec")
    print(f"Subthemes presence: {puncts}")
    assert isinstance(puncts, list), "Result should be a list of binary values"

def test_analyze_empty(analyzer):
    puncts = analyzer.analyze("", "gambling")
    assert puncts == [], "Result should be an empty list"