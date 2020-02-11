import numpy as np
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
	return np.array([random.randint(0, 1)for i in range(n)]).reshape((n,1))


def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

#############################################################################
#RELEASING THE NOIZED VERSION OF DATABASE
#############################################################################

def gen_query(n):
	return np.array([np.array([1]*i + [0]*(n - i)) for i in range(1, n + 1)])

def releasing_dataset(query, dataset):
	return np.array([np.dot(query[i], dataset) + random.randint(0,1) for i in range(len(dataset))])


#############################################################################
#ATTACK WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_no_aux(observation):
	n = len(observation)
	s = [observation[0]] + list((observation[1:] - observation[: -1]))
	# print s
	att = np.array([[0] if r < 0 else [1] if r > 1 else r for r in s] ).reshape(n, 1)
	return att


#############################################################################
#ATTACK2 WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################

def minimize_error_attack_no_aux(query, observation):
	n = len(observation)
	k = 0
	opt, s = 2*n, [observation[0]] + list((observation[1:] - observation[: -1]))
	# print s
	att = np.array([[0] if r < 0 else [1] if r > 1 else r for r in s] ).reshape(n, 1)
	for i in range(n):	
		if s[i] < 0 or s[i] > 1:
			continue
		att[i] = 1 - att[i]
		error = np.sum(abs(np.subtract(observation, np.dot(query, att))) )
		if error < opt:
			opt = error
		else:
			att[i] = 1 - att[i]
		print error, att

	return att

# def minimize_error_attack_no_aux(query, observation):
# 	n = len(observation)
# 	k = 0
# 	opt_att = []
# 	opt, s = 2*n, [observation[0]] + list((observation[1:] - observation[: -1]))
# 	# print s
# 	# att = np.array([[0] if r < 0 else [1] if r > 1 else r for r in s] ).reshape(n, 1)
# 	while k < 20:
# 		k += 1
# 		att = np.array([[random.random()] for _ in observation])
# 		error = np.sum(abs(np.subtract(observation, np.dot(query, att))) )
# 		if error < opt:
# 			opt = error
# 			opt_att = np.round(att)
# 		print opt, opt_att

# 	return opt_att
# def attacks_no_aux(observations):
# 	return [attack_no_aux(o) for o in observations]


# def attacks_no_aux(observations):
# 	return [attack_no_aux(o) for o in observations]


#############################################################################
#ATTACK WITH AUXILLARY INFORMATION AND THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_with_aux(observation, guess):
	attack_no_aux(observation), w
	return

def accuracy(att, data):
	return sum([1 if att[i] == data[i] else 0 for i in range(len(att))])/ (len(data)*1.0)


# def accuracys(atts, datas):
# 	return [accuracy(atts[i], datas[i]) for i in range(len(atts))]

# def testing(datasizes):
# 	datasets = gen_datasets(datasizes)
# 	obs = releasing_datasets(datasets)
# 	atts = attacks_no_aux(obs)
# 	return accuracys(atts, datasets)

def testing_kround(n, k):
	acc = 0.0
	for i in range(k):
		dataset = gen_dataset(n)
		# print dataset
		q = gen_query(n)
		obs = releasing_dataset(q, dataset)
		att = attack_no_aux(obs)
		acc += accuracy(att, dataset)
	return acc/k


def testing_kround_ksize(ns, k):
	return [testing_kround(n, k) for n in ns]

def plot_accuracy(ys, ns):
	plt.figure()
	plt.plot(ns, ys, "ro-", label = "Accuracy.")
	mean = sum(ys) / len(ys)
	plt.plot(ns, [mean]*len(ys), "b-", label = "Average Acc.", linewidth = 3.0)
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
	
	datasizes = gen_datasizes((100, 900),100)  + gen_datasizes((1000,5000), 200) + [50000] # [100, 200, 300] #[100, 500, 1000, 5000] #gen_datasizes((50,600),50)
	# print datasets
	accs = testing_kround_ksize(datasizes, 20)
	# print accs
	plot_accuracy(accs, datasizes)





	

