import numpy
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import math
import scipy
from scipy.stats import beta
from fractions import Fraction
import operator
import time
from matplotlib.patches import Polygon
import statistics
from decimal import *
from scipy.special import gammaln

	#############################################################################
	#GENERATING DATA SIZE AND CONRRESPONDING PARAMETER
	#############################################################################

def gen_dataset(v, n):
	return [int(n * i) for i in v]


def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]


def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]


def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


def discrete_probabilities_from_file(filenames,labels,savename):
	
	#############################################################################
	#READ the prob from file
	#############################################################################

	probabilities_by_steps = []
	steps = []
	for file in filenames:
		f = open(file, "r")
		f.readline()
		s = []
		prob = []
		for line in f:
			l = line.strip("\n").split("&")
			prob.append(float(l[-1]))
			s.append(float(l[-2]))
		probabilities_by_steps.append(prob)
		steps = s

	#############################################################################
	#PLOT the prob within the same bin
	#############################################################################

	plt.figure()
	colors = ["b","r","g"]

	for i in range(len(filenames)):
		plt.plot(steps[-100:], probabilities_by_steps[i][-100:], colors[i], label=(labels[i]))
		# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.xlabel("c / (steps from correct answer, in form of Hellinger Distance)")
	plt.ylabel("Pr[H(BI(x),r) = c]")
	plt.title("Discrete Probabilities")
	plt.legend()
	plt.grid()
	plt.show()




if __name__ == "__main__":

	#############################################################################
	#SETTING UP THE PARAMETERS
	#############################################################################
	datasize = 2
	epsilon = 1.0
	delta = 0.00000001
	prior = dirichlet([1,1])
	dataset = [2,0]

	#############################################################################
	#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
	#############################################################################
	
	datasizes = gen_datasizes((600,600),50)
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	priors = [dirichlet([1,1])] + gen_priors([5,20], 5, 2) + gen_priors([40,100], 20, 2) + gen_priors([150,300], 50, 2) + gen_priors([400,400], 50, 2)

	#############################################################################
	#DO PLOTS BY COMPUTING THE PROBABILITIES FOR GROUP EXPERIMENTS
	#############################################################################

	# for i in range(len(datasizes)):
	# 	row_discrete_probabilities(datasizes[i],epsilon,delta,prior,datasets[i])

	#############################################################################
	#DO PLOTS BY COMPUTING THE PROBABILITIES
	#############################################################################

	row_discrete_probabilities(datasize,epsilon,delta,prior,dataset)

	#############################################################################
	#DO PLOTS BY READING THE PROB FROM FILES
	#############################################################################



	

