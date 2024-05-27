from setuptools import find_packages, setup
import os

here = os.path.abspath(os.path.dirname(__file__))

# with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
#     long_description = f.read()

setup(
    name="bioel",
    version="0.1.0",
    description="An easy-to-use package for all your biomedical entity linking needs.",
    # long_description=long_description,
    url="",
    author="Pathology Dynamics Lab, Georgia Institute of Technology",
    author_email="",
    keywords=[
        "bioel",
        "biomedical",
        "entity",
        "linking",
        "entity-linking",
        "biomedical-entity-linking",
    ],
    packages=find_packages(),
    python_requires=">= 3.9",
    install_requires=[
        "pytest",
        "tqdm",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "obonet",
        "datasets",
        "ujson",
        "bioc",
        "logger",
        "lightning",
        "ipython",
        "cython",
        "transformers",
        "pytorch_transformers",
        "scipy",
        "faiss-gpu",
        "faiss-cpu",
        "wandb",
        "scikit-learn",
        "torch",
    ],
)
