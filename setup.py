from setuptools import setup, find_packages

setup(
    name="textscope",
    version="0.2.0",
    packages=find_packages(),
    package_data={
        'textscope': ['data/config.yaml'],
    },
    install_requires=[
        "torch>=2.6",
        "transformers>=5.3,<6",
        "numpy>=2.0",
        "pyyaml>=6.0.3",
        "nltk>=3.10.0",
    ],
    extras_require={
        "test": ["pytest"],
    },
    license="GNU",  # o la que uses
    license_files=["LICENSE"],
    include_package_data=True,
    python_requires=">=3.10",
    description="A Python text analysis library for relevance and subtheme detection",
    author="Marcos Fernández-Pichel",
    author_email="marcosfernandez.pichel@usc.es",
)
