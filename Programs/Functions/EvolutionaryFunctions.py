# -*- coding: utf-8 -*-

# THIS MODULE CONTAINS ALL THE FUNCTIONS FOR THE EVOLUTIONARY STRATEGIES

import numpy as np
import random
from sklearn.metrics import mean_squared_error as mse
from Functions.GraphAnalysis import degmatrix, lmatrix, eigen, eigen_reduce, eigen_aisle

#----------INDIVIDUAL CREATION FUNCTIONS--------------------------------------------------------------------------------

def graphInd(icls, dim, mask):
    # Function to generate random individuals that satisfy the problem properties.
    # Input: Dimension of the graph you are working with.
    # Output: Random weight matrix which satisfies the patient's properties.
    indGenerator = np.random.rand(dim, dim)
    graphInd = (indGenerator + indGenerator.T) / 2
    np.place(graphInd, mask, 0)
    return icls(graphInd)


def gaussGraphInd(icls, seed, mask):
    # Function to generate gaussian mutated individuals from the top individual of the last optimization. This function
    # is dependent of the custom gaussian mutation function.
    # Input: The top individual as a seed and the mask of the patient.
    # Output: Weight matrix obtained from mutating and individual with good fitness value.
    graphInd = matMutGauss(seed, 0.1, 0.1, mask)
    return icls(graphInd)


#----------MUTATION FUNCTIONS-------------------------------------------------------------------------------------------


def matMutFloat(individual, rowindpb, elemindpb, mask):
    # Function that traverses the individual row by row and element by element depending on a probability.
    # Input: The individual matrix, the probability of mutating a row and the probability of mutating an element.
    # Output: The new mutated individual.
    size = len(individual)
    for i in range(size):
        rowindMut = random.random()
        if rowindMut < rowindpb:
            for j in range(size):
                elemindMut = random.random()
                if elemindMut < elemindpb:
                    attrMut = random.random()
                    individual[i][j], individual[j][i] = attrMut, attrMut
    np.place(individual, mask, 0)
    return individual,


def matMutGauss(individual, rowindpb, elemindpb, mask):
    size = len(individual)
    for i in range(size):
        rowindMut = random.random()
        if rowindMut < rowindpb:
            for j in range(size):
                elemindMut = random.random()
                if elemindMut < elemindpb:
                    mu  = individual[i][j]
                    individual[i][j] = random.gauss(mu, 0.1)
                    while individual[i][j] < 0:
                        individual[i][j] = random.gauss(mu, 0.1)
                    individual[j][i] = individual[i][j]
    np.place(individual, mask, 0)
    return individual,


#----------CROSSOVER FUNCTIONS------------------------------------------------------------------------------------------


def patchCx(ind1, ind2):
    # Crossover function that interchange a symmetric patch of genetic material between individuals, the patch
    #  size depends on a random number.
    # Input: Two individuals matrices.
    # Output: The parents and the children matrices.
    n = len(ind1)
    tam = np.random.randint(1, (n / 2) + 1)
    patch1 = ind1[0:tam,(n-tam):n].copy()
    patch2 = ind2[0:tam, (n - tam):n].copy()
    ind1[0:tam, (n - tam):n], ind1[(n - tam):n, 0:tam] = patch2, patch2.T
    ind2[0:tam, (n - tam):n], ind2[(n - tam):n, 0:tam] = patch1, patch1.T
    del patch1
    del patch2
    return ind1, ind2,


#----------FITNESS FUNCTIONS--------------------------------------------------------------------------------------------


def fit_function(individual, reference):
    degmat = degmatrix(individual)
    lmat = lmatrix(individual, degmat)
    (phys, landas) = eigen(lmat)
    phys_ses = eigen_reduce(phys, landas)
    phy9 = eigen_aisle(phys_ses)
    # Aplicamos el MSE (Error Cuadrático Medio) como función a minimizar
    error_phy = mse(reference, phy9)
    return error_phy,



