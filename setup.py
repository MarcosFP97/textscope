from setuptools import setup, find_packages
import io

setup(
    name="textscope",
    version="0.1.8",
    packages=find_packages(),
    package_data={
        'textscope': ['data/config.yaml'],
    },
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "pyyaml",
        "nltk"
    ],
    extras_require={
        "test": ["pytest"],
    },
    license="GNU",  # o la que uses
    license_files=["LICENSE"],
    include_package_data=True,
    description="A Python text analysis library for relevance and subtheme detection",
    author="Marcos Fernández-Pichel",
    author_email="marcosfernandez.pichel@usc.es",
)
