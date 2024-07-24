from setuptools import setup, find_packages

setup(
    name="textscope",
    version="0.1.5",
    packages=find_packages(),
    package_data={
        'textscope': ['data/config.yaml'],
    },
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pytest",
        "pyyaml",
        "nltk"
    ],
    include_package_data=True,
    description="A Python text analysis library for relevance and subtheme detection",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Marcos Fernández-Pichel",
    author_email="marcosfernandez.pichel@usc.es",
)

