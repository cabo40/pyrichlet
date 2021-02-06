import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyrichlet-cabo40", # Replace with your own username
    version="0.0.1",
    author="Fidel Selva",
    author_email="cfso100@gmail.com",
    description="A package for density estimation and clustering using infinite gaussian mixtures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cabo40/pyrichlet",
    license='GPL3',
    packages=setuptools.find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL3 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)