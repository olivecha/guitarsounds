# guitarsounds

[![pytest](https://github.com/olivecha/guitarsounds/actions/workflows/python-app.yml/badge.svg)](https://github.com/olivecha/guitarsounds/actions/workflows/python-app.yml)

A python package to analyse guitar sounds. Developed as an educational analysis tool with the [Bruand Lutherie School](https://bruand.com/). 
The guitarsound python package documentation is available at [documentation](https://olivecha.github.io/guitarsounds/).

The current release contains usual signal processing tools applied to the analysis of guitar sounds with automated figure generation trough `matplotlib`.

To allow fast and easy analysis of sound files, a graphic user interface is available based on jupyter lab widgets.


## Installation

These steps can be followed to use guitarsound interactively with Jupyter Notebook : 

- Download the Anaconda package management system [link](https://www.anaconda.com/products/distribution).

<img width="1200" alt="Screen Shot 2022-05-22 at 9 18 26 PM" src="https://user-images.githubusercontent.com/78630053/169726111-d80cde25-3630-4cb6-87ac-f487bafa4898.png">

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

<img width="1200" alt="Screen Shot 2022-06-02 at 9 12 01 PM" src="https://user-images.githubusercontent.com/78630053/171768085-c7b3b9e6-3ec9-4c71-b7da-27235a7b3a7a.png">

The Graphic user interface can be launched by typing the following code in a cell of the notebook : 

<img width="1088" alt="Capture d’écran, le 2021-08-29 à 19 01 28" src="https://user-images.githubusercontent.com/78630053/131268136-75835d93-5247-4193-bfc0-e23230adfe79.png">

Pressing shift+enter runs the code in the cell and launches the graphic user interface of guitarsounds

To go further, you may learn the guitarsounds API, see the API Tutorial notebook and the API [documentation](https://olivecha.github.io/guitarsounds/)
