from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["ipython>=6", "nbformat>=4", "requests>=2"]

setup(
    name="bert-crf",
    version="0.0.1",
    author="Dhanachandra",
    author_email="dhana1991@gmail.com",
    description="BERT_CRF model for NER",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Dhana1991/py-project/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: MIT",
    ],
)
