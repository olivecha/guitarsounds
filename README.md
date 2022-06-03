# guitarsounds

[![pytest](https://github.com/olivecha/guitarsounds/actions/workflows/python-app.yml/badge.svg)](https://github.com/olivecha/guitarsounds/actions/workflows/python-app.yml) 

A python package to analyse guitar sounds. Developed as a lutherie research analysis tool with the [Bruand Lutherie School](https://bruand.com/). 
The guitarsound python package documentation is available at [documentation](https://olivecha.github.io/guitarsounds/).

## Motivation

The main goal of this project is to provide a tool to efficiently analyse sound data from research projects in musical instrument desing. While sound analysis packages already exist, they are more directed to feature extraction for machine learning purposes. Additionnaly, some features of interest, like time dependent decay, onset shape and fourier transform peaks distribution are not computable trivially or acurately with existing tools. The current release of the guitarsounds package contains usual and advanced digital signal processing tools applied to the analysis of transient harmonic sounds with easy figure generation trough `matplotlib`. To allow the package functionalities to be used rapidly without learning the API, a graphic user interface is available based on jupyter lab widgets.

## Installation

The following steps can be followed to use guitarsound interactively with Jupyter Notebook. A french version of the installation guide is available [here](https://github.com/olivecha/guitarsounds/wiki/Guide-installation-fran%C3%A7ais).

- Download the Anaconda package management system [link](https://www.anaconda.com/products/distribution).

<img width="900" alt="Screen Shot 2022-05-22 at 9 18 26 PM" src="https://user-images.githubusercontent.com/78630053/169726111-d80cde25-3630-4cb6-87ac-f487bafa4898.png">

- Install the Anaconda package management system ([Tutorial](https://docs.anaconda.com/anaconda/install/windows/)).

- Once Anaconda is installed, the guitarsound package needs to be installed. Without going into heavy details, the guitarsound package is not available from the Anaconda channels and needs to be installed using PIP, the [Package Installer for Python](https://pypi.org/project/guitarsounds/). In order to install a package from PIP in Anaconda, you need to use the Anaconda [Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-prompt-win) :

![win-anaconda-prompt2](https://user-images.githubusercontent.com/78630053/169728612-585f5116-38f4-4642-a1b4-185c702b0151.png)

To install guitarsound, type the following command into the Anaconda Prompt :

```
pip install guitarsounds
```

Once the installation of guitarsounds is finished, the package can be used in the Anaconda Jupyter Notebook environnement. The Jupyter Notebook environnement can be launched from [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/) which is included in the Anaconda package management system. 

![nav-defaults](https://user-images.githubusercontent.com/78630053/169730350-13be0b3e-6851-416c-843d-cbf564ef82b1.png)

Then, navigate to the sub-directory where you want your work to be and create a Jupyter Notebook :

<img width="900" alt="Screen Shot 2022-06-02 at 9 12 01 PM" src="https://user-images.githubusercontent.com/78630053/171768085-c7b3b9e6-3ec9-4c71-b7da-27235a7b3a7a.png">

The graphic user interface can be launched by typing the following code in a cell of the notebook : 

```
import guitarsounds
interface = guitarsounds.Interface()
```

Pressing shift+enter runs the code in the cell and launches the graphic user interface of guitarsounds :

<img width="900" alt="Capture d’écran, le 2021-08-29 à 19 01 28" src="https://user-images.githubusercontent.com/78630053/131268136-75835d93-5247-4193-bfc0-e23230adfe79.png">

To go further, you may learn the guitarsounds API, see the API Tutorial notebook and the API [documentation](https://olivecha.github.io/guitarsounds/)

## Example usage

While extracting quantitative features from sounds allows for a meaningful analysis, listening to them remain an important part of the analysis. Soundfiles ca be loaded by creating a `Sound` class instance with the soundfile path as an argument

```python
mysound = Sound('example_sounds/Wood_Guitar/Wood_E1.wav')
```

The `Sound` instance can then be conditionned to trim it right before its onset and filter the noise if needed:

```python
mysound.condition()
```
The amplitude-time data of the `Sound` instance is stored in a `Signal` class and can be listened in a Jupyter Notebook:

<img width="600" alt="image" src="https://user-images.githubusercontent.com/78630053/171777901-58fad2db-e515-4c1e-ac2a-ca5f3c60708a.png">

Relevant time signal properties can then be rapidly extracted and visualized in the interactive Jupyter Notebook environment. For example, the time damping curve and the associated damping factor is a useful measure when measuring the effects of changes in the guitar design on the decay rate of different notes : 

<img width="400" alt="image" src="https://user-images.githubusercontent.com/78630053/171778130-6892e1c2-b435-4ac4-a7ae-bae289d8fd02.png">

Two different sounds can also be compared using the `SoundPack` class, such as a tangible effect of a design change can be measured. Here we compare the fourier transform peaks of two guitars built using different materials : 

<img width="600" alt="image" src="https://user-images.githubusercontent.com/78630053/171778729-e5e69eff-2ad8-4448-b6fe-d54387d4a6e3.png">

The base API of the guitarsounds classes can also be leveraged to create custom signal analysis features. The following example shows the relatively straightforward implementation of the cumulative fast fourier transform metric from [reference 1](https://arxiv.org/pdf/0901.3708.pdf). 

<img width="600" alt="image" src="https://user-images.githubusercontent.com/78630053/171780702-26ef6aee-8bb1-4561-bbf0-5d28bcacb736.png">

Thus guitarsounds allows both fast and interactive analysis of transient harmonic sounds as well as easily developping and testing new signal analysis features. 

## Community

Feel free to fork the guitarsound repository and submit pull requests implementing usefull changes. If you implement a new feature, please submit associated unit tests in your pull request. Pull requests which pass all the tests and propose changes aligned with the goal of this package should be accepted. 

This project is maintained by the [Bruand Lutherie School](https://bruand.com/), if you have problems with the package or you need support in using it, please write an issue in this repository.
