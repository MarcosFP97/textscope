import pytest
from textscope.subtheme_analyzer import SubthemeAnalyzer

model_name = 'intfloat/multilingual-e5-large-instruct'
# sample_text = 'Buenas, yo soy también enfermo.Como enfermo, conté primeramente con mi familia, la de verdad, la de puertas para adentro. El resto de la familia, parientes. Sin menospreciar su ayuda, su apoyo, su comprensión hacia esta enfermedad, vivo con mi pareja y mis chicos, así que entendíed (entendimos) que era un problema mío que afectaba a nuestro núcleo, así que  era un problema, primordialmente de los que vivimos detrás de esa puerta que pone 5º-A.'
sample_text = 'Perdía el raciocinio apostando cantidades cada vez mayores para sentir estímulos más intensos. He mentido a mi familia.'
# sample_text = 'Apostar en línea es más fácil que nunca, pero la falta de #Regulación adecuada está llevando a índices de #Ludopatía más altos, especialmente en jóvenes. Según la @FADJuventud el 20% de los jóvenes entre 14 y 18 años han apostado en línea.'

@pytest.fixture
def analyzer():
    return SubthemeAnalyzer(model_name)

def test_analyze_sub(analyzer):
    puncts = analyzer.analyze(sample_text)
    print(f"Subthemes scores: {puncts}")
    assert isinstance(puncts, dict), "Results should be a dict of scores"

def test_analyze_subpres(analyzer):
    puncts = analyzer.analyze_bin(sample_text)
    print(f"Subthemes presence: {puncts}")
    assert isinstance(puncts, dict), "Result should be a dict of binary values"

def test_analyze_empty(analyzer):
    puncts = analyzer.analyze("")
    assert puncts == {}, "Result should be an empty dict"