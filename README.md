# TextScope

TextScope is a Python package that helps determine the relevance of a text to predefined profiles of interest and aligns it with specific subthemes. The package is designed to be flexible and configurable via a `config.py` file.

## Installation

You can install TextScope using pip:

```bash
pip install textscope
```

## Configuration


Before using TextScope, define your profiles of interest and subthemes in the config.py file. Example:

```python
THEMES = ['technology', 'AI', 'machine learning', 'software']

SUBTHEMES = {
    '1': 'programming',
    '2': 'data science'
    '3': 'cybersecurity'
}
``` 

## Relevance Analysis

To determine if a text is relevant to any of the predefined profiles:

```python
from textscope.relevance_analyzer import RelevanceAnalyzer

model_name = 'intfloat/multilingual-e5-large-instruct'
text = "This article discusses the latest advancements in AI and machine learning."
analyzer = RelevanceAnalyzer(model_name)
rel_score = analyzer.analyze(text)
print(rel_score)  
``` 

Future versions of the package will include a **filter_corpus** method that is currently under development. 
*NOTE*: We support different embedding models, but we highly recommend to use [e5](https://arxiv.org/abs/2212.03533).

## Subtheme Analysis

To find which subthemes within a profile a text aligns with:

```python
from textscope.subtheme_analyzer import SubthemeAnalyzer

model_name = 'hiiamsid/sentence_similarity_spanish_es'
text = "LA IA es un campo en auge"
analyzer = SubthemeAnalyzer(model_name)
subth_pres = analyzer.analyze_bin(text)
print(subth_pres)  
```

## Testing

To run tests for TextScope, use the following command:

```bash
pytest tests/
```

