import pathlib
import os
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent.resolve()

# The text of the README file
README = (HERE / "README.md").read_text()

# parsing the requirements.txt
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

# This call to setup() does all the work
setup(
    name="guitarsounds",
    version="1.1.0",
    python_requires='>=3',
    description="A python package to analyze and visualize harmonic sounds",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/olivecha/guitarsounds",
    project_urls={
        'Documentation': 'https://olivecha.github.io/guitarsounds/',
    },
    author="Olivier Chabot",
    author_email="olivier.chabot@polymtl.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["guitarsounds"],
    include_package_data=True,
    install_requires=install_requires

)
