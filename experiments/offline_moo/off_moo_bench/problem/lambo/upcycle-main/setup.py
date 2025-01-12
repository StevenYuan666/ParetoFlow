#!/usr/bin/env python
from os import path

from setuptools import find_packages, setup

AUTHOR = "S. Stanton"
NAME = "upcycle"
PACKAGES = find_packages()

REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_PATH = path.join(path.abspath(__file__), REQUIREMENTS_FILE)
with open(REQUIREMENTS_FILE) as f:
    requirements = f.read().splitlines()

setup(
    name=NAME,
    version="0.0.1",
    description="Reusable code snippets",
    author=AUTHOR,
    author_email="ss13641@nyu.edu",
    url="https://github.com/samuelstanton/upcycle",
    install_requires=requirements,
    packages=PACKAGES,
)
