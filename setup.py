import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="guitarsounds",
    version="1.0.0",
    description="Analyze and visualize harmonic sounds",
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
    install_requires=["librosa", "soundfile", "IPython",
                      "matplotlib", "numpy", "noisereduce",
                      "scipy", "tabulate", "ipywidgets"],
    entry_points={
        "console_scripts": [
            "realpython=reader.__main__:main", ]
    },
)
