# DermX experiments
Code for the experiments described in the DermX paper. Requirements must be installed with `python 3.7.5`.

Each of the two architecture folders, `efficientnet` and `resnet` contains the Jupyter notebooks used to run the
hyper-parameter optimisation, the training of the five models considered in the paper, and the code needed to generate 
Grad-CAM visualisations. Grad-CAM notebooks use the methods stored in `explanations/xgradcam.py`.