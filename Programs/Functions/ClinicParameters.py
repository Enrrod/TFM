# -*- coding: utf-8 -*-

# THIS MODULE CONTAINS ALL THE FUNCTIONS TO OBTAIN THE CLINIC PARAMETERS

import numpy as np
from Functions.GraphAnalysis import readgraph, wmatrix


def obtainRef(filedir):
    # Function to the reference eigenvector stored on a txt file in the computer.
    # Input: The route of the txt file.
    # Output: Reference 9th eigenvector for the optimization, array type.
    archivo = open(filedir, "r")
    filas = archivo.readlines()
    archivo.close()
    for i in range(2):
        filas.pop(0)
    for i in range(len(filas)):
        filas[i] = filas[i].rstrip("\n")
    phy_mean = np.empty(0)
    for i in range(len(filas)):
        phy_mean = np.append(phy_mean, float(filas[i]))
    return phy_mean


def patientMask(graphFileDir):
    # Function to obtain the positions of the null values of the patient to create mask with them.
    # Input: Route of the graph file of the patient.
    # Output: Mask of zeros to apply on the random individuals.
    g = readgraph(graphFileDir)
    mask = wmatrix(g)
    mask = np.where(mask > 0, 0, 1)
    return mask