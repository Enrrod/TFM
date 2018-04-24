# -*- coding: utf-8 -*-

# THIS MODULE CONTAINS ALL THE FUNCTIONS USED FOR ANALYZING THE GRAPHS

import numpy as np
from igraph import Graph


def readgraph(x):
	#Function to read from .graphml files.
	#Input: Route of the graph file.
	#Output: Object of type graph.
	g = Graph.Read_GraphML(x)
	return g


def wmatrix(gph):
	# Function to extract the weight matrix of a graph.
	# Input: Object of type graph.
	# Output: Numpy array representing the weight matrix of a graph.
	elist = gph.get_edgelist()
	n = len(elist)
	adj = gph.get_adjacency()
	tam = adj.shape[0]
	wfromgraph = np.zeros((tam, tam))
	lista_pesos = np.empty(0)
	for i in range(n):
		edge = elist[i]
		peso = gph.es[i].attributes()["weight"]
		lista_pesos = np.hstack((lista_pesos, peso))
		ind1 = edge[0]
		ind2 = edge[1]
		wfromgraph[ind1, ind2] = peso
		wfromgraph[ind2, ind1] = peso
	peso_max = np.amax(lista_pesos)
	wfromgraph = wfromgraph / peso_max
	return wfromgraph


def degmatrix(wmat):
	# Function to obtain the degree matrix of a weighted graph.
	# Input: Weight matrix, array type.
	# Output: Numpy array representing the degree matrix of a graph.
	tam = wmat.shape[0]
	d = np.zeros((tam, 1))
	ident = np.identity(tam)
	for i in range(tam):
		for j in range(tam):
			d[i] = d[i] + wmat[i, j]
	degmat = ident * d
	return degmat


def lmatrix(wmat, degmat):
	# Function to obtain the Laplacian matrix of a graph.
	# Input: Weight matrix and degree matrix, array type.
	# Output: Numpy array representing the Laplacian matrix of a graph.
	lmat_nt = degmat - wmat
	lmat_t = lmat_nt.transpose()
	lmat = np.dot(lmat_t, lmat_nt)
	lmat = lmat / 2
	return lmat


def eigen(lmat):
	# Function to obtain the eigenvectors and eigenvalues of the Laplacian matrix of a graph.
	# Input: Laplacian matrix, array type.
	# Output: Vector containing the eigenvalues and matrix containing the eigenvectors, array type.
	(landas, phys) = np.linalg.eig(lmat)
	tam = phys.shape[1]
	index = np.argsort(landas)
	landas = landas[index]
	for i in range(tam):
		phys[i] = phys[i][index]
	return phys, landas


def eigen_reduce(phys, landas):
	# Function to delete from the eigenvector and eigenvalues array the null values repeated.
	# Input: Eigenvector matrix and eigenvalue vector, array type.
	# Output: Matrix containing the eigenvectors without repeating the correspondant to null eigenvalue null,
	# array type.
	del_ind = np.where(landas <= 0)[0]
	landas_ses = np.delete(landas, del_ind)
	phys_ses = np.delete(phys, del_ind, 1)
	return phys_ses


def eigen_aisle(phys_ses):
	# Function to aisle the 9th eigenvector.
	# Input: Eigenvector matrix and eigenvalue vector, array type.
	# Output: 9th eigenvector, array type.
	phy9 = np.empty(0)
	for i in range(phys_ses.shape[0]):
		phy9 = np.hstack((phy9, phys_ses[i, 7].real))
	return phy9