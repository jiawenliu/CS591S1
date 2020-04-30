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
from dirichlet import dirichlet
from dpbayesinfer_Betabinomial import BayesInferwithDirPrior



#############################################################################
#PLOT THE SAMPLING RESULTS BY 4-QUANTILE BOX PLOTS
############################################################################


#############################################################################
#PLOT THE MEAN SAMPLING RESULTS IN SCATTER 
#############################################################################


def plot_mean_error(x,y_list,xstick,xlabel, ylabel, title):
	plt.figure(figsize=(11,8))
	i = 0	
	for i in range(len(y_list)):
		plt.plot(x, y_list[i],'o-',label=ylabel[i])

	plt.xticks(x, xstick, rotation=70,fontsize=12)
	plt.title(title,fontsize=20)
	plt.xlabel(xlabel,fontsize=15)	
	plt.ylabel('Average Hellinger Distance ',fontsize=15)
	plt.grid()
	plt.legend()
	plt.show()

#############################################################################
#SAMPLING UNDER DIFFERENT DATASIZE 
#############################################################################

def accuracy_VS_datasize(epsilon,delta,prior,observations,datasizes):
	data = []
	mean_error = [[],[]]
	for i in range(len(datasizes)):
		observation = observations[i]
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(observation), epsilon, delta, 0.2)
		Bayesian_Model._set_observation(observation)
		print("start" + str(observation))
		Bayesian_Model._experiments(1000)
		print("finished" + str(observation))

		for j in range(len(mean_error)):
			mean_error[j].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[j]])

		# data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		# data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		# data.append(Bayesian_Model._accuracy[Bayesian_Model._keys[4]])
		# a = statistics.median(Bayesian_Model._accuracy[Bayesian_Model._keys[3]])
		# b = statistics.median(Bayesian_Model._accuracy[Bayesian_Model._keys[0]])
		# c = statistics.median(Bayesian_Model._accuracy[Bayesian_Model._keys[4]])


	print('Accuracy / prior: ' + str(prior._alphas) + ", delta: " 
		+ str(delta) + ", epsilon:" + str(epsilon))

	# print mean_error

	plot_mean_error(datasizes, mean_error, datasizes, 
		"Different Datasizes", 
		[r"$Laplace Noise$",
		r"$Geomoetric Noise$"], "")
	
	# plot_error_box(data,"Different Datasizes",datasizes,"Accuracy VS. Data Size",
	# 	[r'$\mathcal{M}^{B}_{\mathcal{H}}$',"LapMech (sensitivity = 2)", "LapMech (sensitivity = 3)"],
	# 	['lightblue', 'navy', 'red'])
	return


def accuracy_VS_datasize(epsilons,delta,prior,observation,datasize):
	data = []
	mean_error = [[],[]]
	for e in epsilons:
		Bayesian_Model = BayesInferwithDirPrior(prior, sum(observation), e, delta, 0.2)
		Bayesian_Model._set_observation(observation)
		print("start" + str(observation))
		Bayesian_Model._experiments(1000)
		print("finished" + str(observation))

		for j in range(len(mean_error)):
			mean_error[j].append(Bayesian_Model._accuracy_mean[Bayesian_Model._keys[j]])

	plot_mean_error(epsilons, mean_error, [round(e, 2) for e in epsilons], 
		"Different Datasizes", 
		[r"$Laplace Noise$",
		r"$Geomoetric Noise$"], "")
	
	return



#############################################################################
#GENERATING DATA SIZE AND CONRRESPONDING PARAMETER
#############################################################################

def gen_dataset(v, n):
	return [int(n * i) for i in v]

def gen_datasets(v, n_list):
	return [gen_dataset(v,n) for n in n_list]

def gen_datasizes(r, step):
	return [(r[0] + i*step) for i in range(0,(r[1] - r[0])/step + 1)]

def gen_priors(r, step, d):
	return [dirichlet([step*i for j in range(d)]) for i in range(r[0]/step,r[1]/step + 1)]


if __name__ == "__main__":

#############################################################################
#SETTING UP THE PARAMETERS
#############################################################################

	datasize = 20
	epsilon = 0.1
	delta = 0.00000001
	prior = dirichlet([1,1])
	data = [500,500]


#############################################################################
#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
#############################################################################
	epsilons = list(numpy.arange(0.01, 0.1, 0.01)) + list(numpy.arange(0.1, 0.5, 0.05))
	datasizes = gen_datasizes((10,50),10) + gen_datasizes((100,500),100) + gen_datasizes((600,1000),200)# + gen_datasizes((1000,5000),1000)#[300] #[8,12,18,24,30,36,42,44,46,48]#,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80]
	percentage = [0.5,0.5]
	datasets = gen_datasets(percentage, datasizes)
	
#############################################################################
#DOING PLOTS OF ACCURACY V.S. THE DATA SIZE
#############################################################################
	
	accuracy_VS_datasize(epsilons,delta,prior,data,1000)



	

