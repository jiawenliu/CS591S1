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
	return [sum(dataset[:i]) + random.randint(0,1) for i in range(len(dataset))]

def releasing_datasets(datasets):
	return [releasing_dataset(d) for d in datasets]


#############################################################################
#ATTACK WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_no_aux(observation):
	observation = [0] + observation
	return [observation[i] - observation[i - 1] for i in range(1, len(observation))]


def attacks_no_aux(observations):
	return [attack_no_aux(o) for o in observations]


#############################################################################
#ATTACK WITH AUXILLARY INFORMATION AND THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_with_aux(observation):
	return

def accuracy(att, data):
	return sum([1 if att[i] == data[i] else 0 for i in range(len(att))])/ (len(att)*1.0)


def accuracys(atts, datas):
	return [accuracy(atts[i], datas[i]) for i in range(len(atts))]

def plot_accuracy(labels):
	
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
	#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
	#############################################################################
	
	datasizes = [100, 500, 1000] #[100, 500, 1000, 5000] #gen_datasizes((50,600),50)
	datasets = gen_datasets(datasizes)
	obs = releasing_datasets(datasets)
	atts = attacks_no_aux(obs)
	print accuracys(obs, datasets)
	# print datasets




	

