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
	return np.array([random.randint(0, 1)for i in range(n)])


def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

#############################################################################
#RELEASING THE NOIZED VERSION OF DATABASE
#############################################################################

def gen_query(n):
	return np.array([np.array([1]*i + [0]*(n - i)) for i in range(1, n + 1)])

def releasing_dataset(query, dataset):
	return np.array([np.dot(query[i], dataset.transpose()) + random.randint(0,1) for i in range(len(dataset))])


#############################################################################
#ATTACK WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_no_aux(observation):
	# print ":attack:"
	# print observation
	rec_counter = [observation[0] - 1 if observation[0] > 1 else observation[0]]
	# print rec_counter
	for i in range(1, len(observation)):
		s = observation[i] - rec_counter[i - 1]
		if s > 1:
			rec_counter.append(observation[i] - 1)
		elif s < 0:
			rec_counter.append(observation[i])
			j = i
			while j - 1 >= 0 and rec_counter[j] < rec_counter[j - 1]:
				j -= 1
				rec_counter[j] -= 1
		else:
			rec_counter.append(observation[i])
		# print rec_counter

	# print ([rec_counter[i] - rec_counter[i - 1] for i in range(1, len(observation))])
	return np.array([rec_counter[0]] + [rec_counter[i] - rec_counter[i - 1] for i in range(1, len(observation))])


#############################################################################
#ATTACK WITH AUXILLARY INFORMATION AND THE OBSERVATION OF ONE DATABASE
#############################################################################
def attack_with_aux(observation, guess):
	rec_counter = [observation[0] - 1 if observation[0] > 1 else observation[0]]
	# print rec_counter
	for i in range(1, len(observation)):
		s = observation[i] - rec_counter[i - 1]
		if s > 1:
			rec_counter.append(observation[i] - 1)
		elif s < 0:
			rec_counter.append(observation[i])
			j = i
			while j - 1 >= 0 and rec_counter[j] < rec_counter[j - 1]:
				j -= 1
				rec_counter[j] -= 1
		else:
			rec_counter.append(observation[i] if s == guess[i] else rec_counter[i - 1] + guess[i])
		# print rec_counter

	# print ([rec_counter[i] - rec_counter[i - 1] for i in range(1, len(observation))])
	return np.array([rec_counter[0]] + [rec_counter[i] - rec_counter[i - 1] for i in range(1, len(observation))])

def accuracy(att, data):
	return sum([1 if att[i] == data[i] else 0 for i in range(len(att))])/ (len(data)*1.0)


def exprmt_k(n, k):
	acc = 0.0
	for i in range(k):
		# print "round:", i
		dataset, q = gen_dataset(n), gen_query(n)
		# print "data:", dataset
		# acc += accuracy(attack_no_aux(releasing_dataset(q, dataset)), dataset)
		guess = [d if random.random() >= (1.0/3) else 1 - d for d in dataset]
		acc += accuracy(attack_with_aux(releasing_dataset(q, dataset), guess), dataset)
	return acc/k


def exprmt_k_ns(ns, k):
	return [exprmt_k(n, k) for n in ns]

def plot_accuracy(ys, ns):
	plt.figure()
	plt.plot(ns, ys, "ro-", label = "Accuracy.")
	plt.plot(ns, [sum(ys) / len(ys)]*len(ys), "b-", label = "Average Acc.", linewidth = 3.0)
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
	
	datasizes = [1000] # gen_datasizes((100, 900),100) # + gen_datasizes((1000,5000), 200)# + [50000] # [100, 200, 300] #[100, 500, 1000, 5000] #gen_datasizes((50,600),50)
	plot_accuracy(exprmt_k_ns(datasizes, 100), datasizes)





	

