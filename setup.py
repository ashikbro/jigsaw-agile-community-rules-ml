from setuptools import find_packages, setup

setup(
    name="jigsaw-ml",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.2",
        "spacy>=3.1.0",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "tqdm>=4.62.2",
        "python-dotenv>=0.19.0",
        "scikit-optimize>=0.9.0",
        "sentence-transformers>=2.2.0",
    ],
)
