"""
Paretoflow is a Python package for offline multi-objective optimization using
Generative Flow Models with Multi Predictors Guidance to approximate the Pareto front.
"""

from setuptools import setup

__version__ = "0.1.4"
__author__ = "Ye Yuan, Can Chen"

setup(
    name="paretoflow",
    version=__version__,
    description="Paretoflow is a Python package for offline multi-objective optimization using \
    Generative Flow Models with Multi Predictors Guidance to approximate the Pareto front.",
    url="https://github.com/StevenYuan666/ParetoFlow",
    author=__author__,
    author_email="ye.yuan3@mail.mcgill.ca",
    license="MIT License",
    packages=["paretoflow"],
    python_requires=">=3.9",
    keywords="optimization",
    install_requires=[
        "pymoo>=0.6.0",
        "numpy>=1.23.2",
        "scipy>=1.10.1",
        "torch>=2.0.1",
        "tqdm>=4.66.5",
    ],
    platforms="any",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    long_description=open("long_description.md", "r").read(),
    long_description_content_type="text/markdown",
)
