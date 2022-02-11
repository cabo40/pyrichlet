#!/usr/bin/env python

# Copyright (C) 2020-2021 Fidel Selva
# License: Apache License 2.0

import setuptools
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyrichlet",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Fidel Selva",
    author_email="cfso100@gmail.com",
    description="A package for density estimation and clustering using "
                "infinite Gaussian mixtures with stick-breaking weighting "
                "structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cabo40/pyrichlet",
    project_urls={
        'Bug Tracker': 'https://github.com/cabo40/pyrichlet/issues',
        'Documentation': 'https://pyrichlet.readthedocs.io',
        'Source Code': 'https://github.com/cabo40/pyrichlet'
    },
    license='Apache License, Version 2.0',
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn'
    ],
    extras_require={
        "tqdm": ["tqdm"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
