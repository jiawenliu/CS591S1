import numpy as np
import matplotlib.pyplot as plt
import random
import math
from functools import partial

#############################################################################
#GENERATING DATA SIZE AND CONRRESPONDING PARAMETER
#############################################################################

def gen_data(d, n):
	return np.array([[random.choice([-1,1]) for _ in range(d)] for i in range(n)])

def gen_datas_n(n_list, d):
	return [gen_data(d, n) for n in n_list]

def gen_datas_d(n, d_list):
	return [gen_data(d, n) for d in d_list]


def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

def gen_cfft(d):
	return np.array([random.choice([-1/sqrt(d), 1/sqrt(d)]) for _ in range(d)])
#############################################################################
#RELEASING THE NOIZED VERSION OF QUERY
#############################################################################

def F_test(A, X):
	return np.dot(A, X.transpose())



def query_avg(data):
	return np.array(map(sum, data.transpose()))/(len(data)*1.0)

def gaussi_mech(query, sigma, data):
	return query(data) + np.array([random.gauss(0, sigma) for _ in data[0]]) 

def rounding_mech(query, sigma, data):
	return np.array(map(round, query(data)/sigma)) * sigma
#############################################################################
#ATTACK WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################

def scores(A, data):
	return F_test(A, data)

#############################################################################
#ATTACK2 WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################

def true_positive(sc1, sc2):

	return sum(map(lambda s: 1.0 if s*1.0/len(sc1) > 0.95 else 0, 
		[sum(map(lambda s2: 1.0 if s1 > s2 else 0, sc2)) for s1 in sc1])) / len(sc1)


def exprmt(d, n, sigma):
	p, k = 0.0, 1
	for i in range(k):
		data, out_data = gen_data(d, n), gen_data(d, n)
		A = gaussi_mech(query_avg, sigma, data)
		s_in, s_out = scores(A, data), scores(A, out_data)
		p += true_positive(s_in, s_out)

	return p/k


def exprmt_n(n_list, d, sigma):
	return [exprmt(d, n, sigma) for n in n_list]

def exprmt_d(n, d_list, sigma):
	return [exprmt(d, n, sigma) for d in d_list]

def plot_accuracy(ys, ns):
	plt.figure()
	plt.plot(ns, ys, "ro-")
	plt.xlabel("d / dimension of the database (# of attributes)")
	plt.ylabel("true positive rate")
	# plt.title("Linear Attack")
	plt.legend()
	plt.grid()
	plt.show()


if __name__ == "__main__":


	#############################################################################
	#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
	#############################################################################
	d_list = [100, 200, 400, 800, 2000, 5000]
	tp = exprmt_d(100, d_list, 1/3.0)
	plot_accuracy(tp, d_list)





	

