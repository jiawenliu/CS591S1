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

def gen_dataset(n):
	return [random.randint(0, 1)for i in range(n)]


def gen_datasets(n_list):
	return [gen_dataset(n) for n in n_list]


def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

#############################################################################
#RELEASING THE NOIZED VERSION OF DATABASE
#############################################################################

def releasing_dataset(dataset):
	return [sum(dataset[:(i + 1)]) + random.randint(0,1) for i in range(len(dataset))]

def releasing_datasets(datasets):
	return [releasing_dataset(d) for d in datasets]


#############################################################################
#ATTACK WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_no_aux(observation):
	observation = [0] + observation
	att = []
	for i in range(1, len(observation)):
		r = (observation[i] - observation[i - 1])
		if r < 0:
			r = 0
		elif r > 1:
			r = 1
		att.append(r)
	return att


def attacks_no_aux(observations):
	return [attack_no_aux(o) for o in observations]


#############################################################################
#ATTACK WITH AUXILLARY INFORMATION AND THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_with_aux(observation):

	return attack_no_aux(observation), w

def accuracy(att, data):
	# print att
	return sum([1 if att[i] == data[i] else 0 for i in range(len(att))])/ (len(att)*1.0)


def accuracys(atts, datas):
	return [accuracy(atts[i], datas[i]) for i in range(len(atts))]

def plot_accuracy(ys, ns):
	
	#############################################################################
	#PLOT the prob within the same bin
	#############################################################################

	plt.figure()
	# colors = ["b","r","g"]

	plt.plot(ns, ys)
		# plt.plot(T, approximate_bounds, 'g^', label=('Expmech_SS zApproximate Bound'))
	plt.xlabel("n / size of the database")
	plt.ylabel("accuracy / fraction of the bits recovered")
	plt.title("Linear Attack")
	plt.legend()
	plt.grid()
	plt.show()




if __name__ == "__main__":


	#############################################################################
	#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
	#############################################################################
	
	datasizes = gen_datasizes((100,1000),20) + gen_datasizes((1000,5000),100) # [100, 200, 300] #[100, 500, 1000, 5000] #gen_datasizes((50,600),50)
	datasets = gen_datasets(datasizes)
	obs = releasing_datasets(datasets)
	# print datasets
	# print obs
	atts = attacks_no_aux(obs)
	acs = accuracys(atts, datasets)
	plot_accuracy(acs, datasizes)
	# print datasets




	

