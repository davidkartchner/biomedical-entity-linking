from setuptools import setup, find_packages, Extension
import os
import subprocess

try:
    from Cython.Build import cythonize
except ImportError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "cython"])
    from Cython.Build import cythonize

try:
    import numpy
except ImportError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "numpy"])
    import numpy

# Define the Cython extension
cython_extensions = [
    Extension(
        "bioel.models.arboel.biencoder.model.special_partition.special_partition",
        ["bioel/models/arboel/biencoder/model/special_partition/special_partition.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

# Setup function to include the Cython extension
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
    python_requires=">=3.9",
    setup_requires=["numpy", "cython"],  # Ensures numpy and cython are installed early
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
    ext_modules=cythonize(cython_extensions),
    include_package_data=True,
    zip_safe=False,
)
