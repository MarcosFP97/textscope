# TextScope :books::mag:

![PyPI - Downloads](https://img.shields.io/pypi/dm/textscope)

*TextScope* is a Python package that helps determine the relevance of a text to predefined profiles of interest and aligns it with specific subthemes. The package is designed to be flexible and configurable via a `config.py` file.

## Installation

You can install *TextScope* using pip:

```bash
pip install textscope
```

## Configuration

Before using *TextScope*, define your profiles of interest and subthemes in the config.py file. Example:

```python
THEMES = ['technology', 'AI', 'machine learning', 'software']

SUBTHEMES = {
    '1': 'Natural Language Processing',
    '2': 'Transfomer-based architecture'
    '3': 'Computer Vision and multimodality'
}
``` 

In this `config` example, we defined a series of themes or keyphrases related to AI. They can be used in combination with the relevance filter to keep only highly on-topic documents. We also defined a series of subthemes to determine whether the analyzed text discuss the subtheme or not.

## Relevance Analysis

To determine if a text is relevant to any of the predefined *themes* or *profiles*:

```python
from textscope.relevance_analyzer import RelevanceAnalyzer

model_name = 'intfloat/multilingual-e5-large-instruct'
text = "This article discusses the latest advancements in AI and machine learning."
analyzer = RelevanceAnalyzer(model_name)
rel_score = analyzer.analyze(text)
print(rel_score)  ## it will return a high score of relevance for the themes (> 86.)
``` 
One possible application  of this method would be to filter out texts that are not highly relevant to the topic. Future versions of the *TextScope* will include a *filter_corpus* method that will remove the out-of-scope texts from a corpus (currently *under development*). 
**NOTE**: *TextScope* is agnostic to the embedding model underneath, but we highly recommend to use [e5](https://huggingface.co/intfloat/multilingual-e5-large-instruct) multilingual instruct version.It is highly flexible and accepts instructions in natural language.

The **default** config file provided with *TextScope* defines a profile of interest in <ins>Spanish</ins> related to <ins>pathological gambling</ins> and a list of subthemes representing symptoms of the pathology. It is an example of the **multilingual** support of this package and its application to **complex real scenarios**.

## Subtheme Analysis

This class allows to test whether a text discuss or not the subthemes defined in the `config`:

```python
from textscope.subtheme_analyzer import SubthemeAnalyzer

model_name =  intfloat/multilingual-e5-large-instruct'
text = "Transformer-based architecture is the state-of-the-art in NLP."
analyzer = SubthemeAnalyzer(model_name)
subth_pres = analyzer.analyze_bin(text) # default threshold set to 86.
print(subth_pres)  # For this sentence and subthemes it should output {'1':1, '2':1, '3':0}
```
If we do not want a binary output, we also provide a method that outputs the similarity:
```python
from textscope.subtheme_analyzer import SubthemeAnalyzer

model_name =  intfloat/multilingual-e5-large-instruct'
text = "Transformer-based architecture is the state-of-the-art in NLP."
analyzer = SubthemeAnalyzer(model_name)
subth_prob = analyzer.analyze(text) # default threshold set to 86.
print(subth_prob)  # For this sentence and subthemes it should output {'1':1, '2':1, '3':0}
```

## Testing

To run tests for TextScope, use the following command:

```bash
pytest -s tests/
```

