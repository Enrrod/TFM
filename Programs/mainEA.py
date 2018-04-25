# -*- coding: utf-8 -*-

# -----IMPORTACIÓN DE LOS MÓDULOS NECESARIOS----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms
from Functions.GraphAnalysis import readgraph, wmatrix, degmatrix, lmatrix, eigen, eigen_reduce, eigen_aisle
from Functions.EvolutionaryFunctions import graphInd, gaussGraphInd, matMutFloat, matMutGauss, patchCx, fit_function
from Functions.ClinicParameters import obtainRef, patientMask


# -----DEFINICIÓN DE LOS PARÁMETROS CLÍNICOS (AUTOVECTOR SANO Y CONEXIONES DAÑADAS)-------------------------------------

filedir = '/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Healthy/60nodos/mean_values.txt'
phy_mean = obtainRef(filedir)
graphFileDir = '/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Issues/60nodos/opt_1.graphml'
mask = patientMask(graphFileDir)

# -----DEFINICIÓN DE LOS PARÁMETROS DE DEAP PARA FASE 1-----------------------------------------------------------------

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', np.ndarray, fitness=creator.FitnessMin)

# Registro del individuo y la población en la toolbox

toolbox = base.Toolbox()
toolbox.register('individual', graphInd, creator.Individual, dim=70, mask=mask)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# Registro de las estrategias evolutivas

toolbox.register('evaluate', fit_function, reference=phy_mean)
toolbox.register('mate', patchCx)
toolbox.register('mutate', matMutFloat, rowindpb=0.1, elemindpb=0.1, mask=mask)
toolbox.register('select', tools.selTournament, tournsize=3)

# -----EJECUCIÓN DE LA FASE 1 DEL ALGORITMO-----------------------------------------------------------------------------

# Definición de parametros del algoritmo

mutpb = 0.3
cxpb = 0.3

print("Cxpb= " + str(cxpb) + " y mutpb= " + str(mutpb))

# Creación de la estadisticas que se van a registrar

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min, axis=0)
stats.register('avg', np.mean, axis=0)
logbook = tools.Logbook()

# Definición del tamaño de la población y el número de generaciones de la primera fase

population = toolbox.population(100)
NGEN = 500
hof = tools.HallOfFame(1, similar=np.array_equal)

# Ejecución de la primera fase

for gen in range(NGEN):
	offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
	fits = toolbox.map(toolbox.evaluate, offspring)
	for fit, ind in zip(fits, offspring):
		ind.fitness.values = fit
	population = toolbox.select(offspring, k=len(population))
	top = tools.selBest(population, k=1)
	record = stats.compile(population)
	logbook.record(gen=gen, **record)
	print("Generación " + str(gen + 1) + " de la primera fase completada")


print("Fin de la primera fase de optimización")
print("Configurando parámetros para la segunda fase...")

# -----DEFINICIÓN DE LOS PARÁMETROS DE DEAP PARA FASE 2-----------------------------------------------------------------

# Individuo semilla a partir del que crear la poblacion incial de la segunda fase

seed = top[0]

# Registro del individuo y la población en la toolbox

toolbox.register('individual', gaussGraphInd, creator.Individual, seed=seed, mask=mask)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# Registro de las estrategias evolutivas

toolbox.register('evaluate', fit_function, reference=phy_mean)
toolbox.register('mate', patchCx)
toolbox.register('mutate', matMutGauss, rowindpb=0.1, elemindpb=0.1, mask=mask)
toolbox.register('select', tools.selTournament, tournsize=3)

# Definición del tamaño de la población y el número de generaciones de la segunda fase

population = toolbox.population(100)
NGEN2 = NGEN * 2
hof = tools.HallOfFame(1, similar=np.array_equal)

# Ejecución de la segunda fase

for gen in range(NGEN,NGEN2):
	offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
	fits = toolbox.map(toolbox.evaluate, offspring)
	for fit, ind in zip(fits, offspring):
		ind.fitness.values = fit
	population = toolbox.select(offspring, k=len(population))
	top = tools.selBest(population, k=1)
	record = stats.compile(population)
	logbook.record(gen=gen, **record)
	print("Generación " + str(gen + 1) + " de la segunda fase completada")

# -----OBTENCIÓN DE ESTADÍSTICAS Y REPRESENTACIÓN GRÁFICA---------------------------------------------------------------

# Creación de variables para guardar las estadísticas

generation = logbook.select('gen')
fitness_min = logbook.select('min')
fitness_avg = logbook.select('avg')
top_ind = top[0]

# Hacemos cálculos para dar valores relativos respecto al estado inicial del paciente

g = readgraph(graphFileDir)
w_pat = wmatrix(g)
init_error = fit_function(w_pat, phy_mean)
init_error = init_error[0]
fitness_rel = (fitness_min / init_error) * 100
fitness_avg_rel = (fitness_avg / init_error) * 100

# Creación de los elementos de la gráfica

plt.figure()
line1 = plt.plot(generation, fitness_rel, "b-", label="Relative fitness")
line2 = plt.plot(generation, fitness_avg_rel, "r-", label="Relative average Fitness")
plt.axis([0, len(generation), -10, 100])
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title('Eigenvalue Optimization')
lines1 = [line1[0], line2[0]]
labs1 = [line1[0].get_label(), line2[0].get_label()]
plt.legend(lines1, labs1, loc="upper right")
plt.title("cxpb= " + str(cxpb) + " mutpb= " + str(mutpb))
plt.show()

# -----ALMACENAMIENTO DE ESTADÍSTICAS-----------------------------------------------------------------------------------

# Definición de las rutas de los archivos a guardar

'''
minFitFile = "/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Issues/AGtest/cxpb" + str(cxpb) + "_mutpb" + str(mutpb) + "/fitness_min" + str(
	i + 1) + ".csv"
relMinFitFile = "/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Issues/AGtest/cxpb" + str(cxpb) + "_mutpb" + str(mutpb) + "/fitness_rel" + str(
	i + 1) + ".csv"
avgFitFile = "/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Issues/AGtest/cxpb" + str(cxpb) + "_mutpb" + str(mutpb) + "/fitness_avg" + str(
	i + 1) + ".csv"
relAvgFitFile = "/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Issues/AGtest/cxpb" + str(cxpb) + "_mutpb" + str(mutpb) + "/fitness_avg_rel" + str(
	i + 1) + ".csv"
topIndFile = "/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Issues/AGtest/cxpb" + str(cxpb) + "_mutpb" + str(mutpb) + "/top_ind" + str(
	i + 1) + ".csv"
plotFile = "/home/enrique/Dropbox/TFM/grafos/DrugsCompare/Issues/AGtest/cxpb" + str(cxpb) + "_mutpb" + str(mutpb) + "/plot" + str(
	i + 1) + ".pdf"

# Guardamos los archivos en las rutas especificadas

np.savetxt(minFitFile, fitness_min, delimiter=",")
np.savetxt(relMinFitFile, fitness_rel, delimiter=",")
np.savetxt(avgFitFile, fitness_avg, delimiter=",")
np.savetxt(relAvgFitFile, fitness_avg_rel, delimiter=",")
np.savetxt(topIndFile, top_ind, delimiter=",")
plt.savefig(plotFile)
'''