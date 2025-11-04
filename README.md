Coupled-Time-Crystals

Codes for reproducing the data and figures of the article entitled "Thermodynamics and energy storage in coupled time crystals"

Authors: Paulo J. Paulino,Albert Cabot, Gabriele De Chiara, Mauro Antezza, Igor Lesanovsky, and Federico Carollo

Arxiv: https://arxiv.org/abs/2411.04836

Basic Information about the Directories:

The codes are written in Python.
The codes require NumPy, SciPy, QuTiP, and Matplotlib.

The "figures" folder contains a file .svg, where the figures were post-processed (adjusted font size, font style, legend size, etc.).

There are two main codes, resposible for solving the mean-field and correlations set of equations. 

They files are named MeanField.py and Correlations.py

Then, we dived the codes in two folders, one for each setup. Inside each folder there are two folders, one for the codes in the main text and the other for the results within the appendix. The data and the plots are also in separed folders. 

The folder structure is

- Main 
    - figures 
    - codes
        - Setup 1 
            - MainText 
                - Figures
                - Data
            - Appendix
                - Figures
                - Data
        - Setup 2
            - MainText 
                - Figures
                - Data
            - Appendix
                - Figures
                - Data





