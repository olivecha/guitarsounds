import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent.resolve()

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="guitarsounds",
    version="1.0.0",
    description="A python package to analyze and visualize harmonic sounds",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/olivecha/guitarsounds",
    author="Olivier Chabot",
    author_email="olivier.chabot.2@ens.etsmtl.ca",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["guitarsounds"],
    include_package_data=True,
    install_requires=["soundfile", "IPython",
                      "matplotlib", "numpy", "noisereduce",
                      "scipy", "tabulate", "ipywidgets"],

)