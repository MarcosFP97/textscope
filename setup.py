from setuptools import setup, find_packages

setup(
    name="textscope",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pytest",
        "sentence-transformers",
        "nltk"
    ],
    include_package_data=True,
    description="A text analysis library for relevance and subtheme detection",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Marcos Fernández-Pichel",
    author_email="marcosfernandez.pichel@usc.es",
)

