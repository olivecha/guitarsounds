# guitarsounds

A python package to analyse guitar sounds. Developed as an educational analysis tool with the [Bruand Lutherie School](https://bruand.com/). 
The API documentation is available at [documentation](https://olivecha.github.io/guitarsounds/).

The current release contains usual signal processing tools applied to guitar sounds with automated figure generation trough `matplotlib` usable with a jupyter notebook GUI.

The analysis tools also feature a "frequency bin separation" tool where the sound is decomposed into 6 bins ("bass", "mid", "highmid", "uppermid", "presence" and "brillance"). This analysis should provide luthiers a more intuitional insight on how a guitar behave trough the frequency range, while retaining transient time information. 

## Installing the package

The `guitarsounds` package is installable trough `pip`:

```
pip install guitarsounds
```

## Using the Jupyter Notebook GUI

1. Open the jupyter notebook console in your current environement. 
2. Create a new notebook
3. Type the following code in the first cell

<img width="1088" alt="Capture d’écran, le 2021-08-29 à 19 01 28" src="https://user-images.githubusercontent.com/78630053/131268136-75835d93-5247-4193-bfc0-e23230adfe79.png">

4. Press shift + enter

To go further, you may learn the guitarsounds API, see the API Tutorial notebook and the API documentation.
