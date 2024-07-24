# TextScope 📖🔍

![PyPI - Downloads](https://img.shields.io/pypi/dm/textscope)

*TextScope* is a Python package that helps determine the relevance of a text to predefined profiles of interest and aligns it with specific subthemes. The package is designed to be flexible and configurable via a `config.yaml` file. Features:

- Multilingual support 🗣️
- Powered by Transformers techonology 🤖
- Easily customizable for other profiles of interest ⚙️🔧

## Installation

You can install *TextScope* using pip (🐍):

```bash
pip install textscope
```

## Configuration

### Default Configuration 
By default, *TextScope* looks for a configuration file included in the package. You can create your own configuration file if you need to customize the profiles and subthemes. The default *config.yaml* included with the package provides the following profiles:

- **gambling** 🎲 in Spanish. Subthemes are DSM-V questionnaire items.
- **mental_health** 🧠 in English. Subthemes are BDI-II questionnaire items.
- **climat** 🌦️ in French. Subthemes related to climate change phenomena.

You can find more info in the `textscope/data/config.yaml` file.

### Custom Configuration

If you want to use a custom configuration file, you can specify the file path in your code:

```python
from textscope.config_loader import load_config

# Load custom configuration
load_config('path/to/custom_config.yaml')
```
Example of custom `config.yaml`:

```yaml
PROFILES:
    ai: ['technology', 'AI', 'machine learning', 'software']

SUBTHEMES:
    ai: ['Natural Language Processing', 'Transfomer-based architecture', 'Computer Vision and multimodality']
```

## Relevance Analysis

To determine if a text is relevant to any of the predefined *profiles*. One possible application  of this method would be to filter out texts that are not highly relevant to the topic. Future versions of the *TextScope* will include a *filter_corpus* method that will remove the out-of-scope texts from a corpus (currently *under development*). 

### Default Configuration

```python
from textscope.relevance_analyzer import RelevanceAnalyzer

text = "La adicción al juego es una enfermedad, pero es la única enfermedad que te puede hacer rico. La artritis no te va a hacer ganar un centavo"
profile = 'gambling'

analyzer = RelevanceAnalyzer()
rel_score = analyzer.analyze(text, profile)
print(rel_score)  ## it will return a high score of relevance for the profile (> 86.)
```

### Custom Configuration

```python
from textscope.config_loader import load_config
from textscope.relevance_analyzer import RelevanceAnalyzer

load_config('path/to/custom_config.yaml')

# Realizar un análisis de relevancia
text = "Transformers based architecture is the sota in NLP."
profile = 'ai'

relevance_analyzer = RelevanceAnalyzer()
is_relevant = relevance_analyzer.analyze(text, profile)
print(rel_score)  ## it will return a high score of relevance for the profile (> 86.)
```

## Subtheme Analysis

This class allows to test whether a text discuss or not the subthemes defined in the profile.

### Default Configuration

```python
from textscope.subtheme_analyzer import SubthemeAnalyzer

text = 'Perdía el raciocinio apostando cantidades cada vez mayores para sentir estímulos más intensos. He mentido a mi familia.'
profile = 'gambling'

analyzer = SubthemeAnalyzer()
subth_pres = analyzer.analyze_bin(text, profile) # default threshold set to 86.
print(subth_pres)  # For this sentence and subthemes it should output [0, 1, 0, 0, 0, 1, 0, 0, 0]
```

### Custom Configuration

```python
from textscope.config_loader import load_config
from textscope.subtheme_analyzer import SubthemeAnalyzer

load_config('path/to/custom_config.yaml')
text = "Transformer-based architecture is the state-of-the-art in NLP."
profile = 'ai'

analyzer = SubthemeAnalyzer()
subth_pres = analyzer.analyze_bin(text, profile) # default threshold set to 86.
print(subth_pres)  # For this sentence and subthemes it should output [1,1,0]
```
If we do not want a binary output, we also provide a method that outputs the similarity:

```python
from textscope.subtheme_analyzer import SubthemeAnalyzer

text = 'Perdía el raciocinio apostando cantidades cada vez mayores para sentir estímulos más intensos. He mentido a mi familia.'
profile = 'gambling'

analyzer = SubthemeAnalyzer()
subth_scoring = analyzer.analyze(text, profile) # default threshold set to 86.
print(subth_scoring)  # For this sentence and subthemes it should output [82.50125885009766, 87.5889663696289, 82.89108276367188, 81.27981567382812, 84.01229095458984, 86.728271484375, 82.63910675048828, 82.18984985351562, 82.15728759765625]
```

## Testing

To run tests for TextScope, use the following command:

```bash
pytest -s tests/
```

## Collaborate

This is an under development project, **PR** are welcome and feel free to contact me at `marcosfernandez.pichel@usc.es`.


