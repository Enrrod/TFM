# -*- coding: utf-8 -*-

# THIS MODULE CONTAINS ALL THE FUNCTIONS OF THE EVOLUTIONARY STRATEGIES

import numpy as np
from Functions.GraphAnalysis import readgraph, wmatrix


def obtainRef(filedir):
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
    g = readgraph(graphFileDir)
    mask = wmatrix(g)
    mask = np.where(mask > 0, 0, 1)
    return mask