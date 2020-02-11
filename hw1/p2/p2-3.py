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
	return np.array([random.choice([-1/math.sqrt(d), 1/math.sqrt(d)]) for _ in range(d)])



def gen_label(data, c, sigma):
	return np.dot(c, data.transpose()) + np.array([random.gauss(0, sigma) for _ in data]) 

#############################################################################
#RELEASING THE NOIZED VERSION OF QUERY
#############################################################################


def LSLR(data, label):
	return np.linalg.lstsq(data, label)[0]

def classifier(w, target):
	return 1.0 if abs(np.dot(w, target[0].transpose()) - target[1]) < 0.01 else 0.0

def scores(w, pos_targets, neg_targets):
	return [classifier(w, p) for p in pos_targets], [classifier(w, p) for p in neg_targets]

#############################################################################
#ATTACK2 WITH ONLY THE KNOWLEDGE OF THE OBSERVATION OF ONE DATABASE
#############################################################################

def true_positive(sc1, sc2):
	return sum(sc1) / (len(sc1))

def false_negtive(sc1, sc2):
	return 1.0 - sum(sc1) / (len(sc1))

def true_negtive(sc1, sc2):
	return sum(sc2) / (len(sc2))

def false_positive(sc1, sc2):
	return 1.0 - sum(sc2) / (len(sc2))


def exprmt(d, n, sigma):
	data, coefft, neg_data = gen_data(d, n), gen_cfft(d), gen_data(d, n)
	y, neg_y = gen_label(data, coefft, sigma), gen_label(neg_data, coefft, sigma)
	score = scores(LSLR(data, y), zip(data, y), zip(neg_data, neg_y))
	return true_positive(score[0], score[1]), true_negtive(score[0], score[1])


def exprmt_n(n_list, d, sigma):
	return [exprmt(d, n, sigma) for n in n_list]

def exprmt_d(n, d_list, sigma):
	return [exprmt(d, n, sigma) for d in d_list]

def plot_accuracy(ys, ns):
	plt.figure()
	plt.plot(ns, [y[0] for y in ys], "ro-", label = "true positive rate")
	plt.plot(ns, [y[1] for y in ys], "bo-", label = "true negtive rate")
	plt.xlabel("d / dimension of the database (# of attributes)")
	plt.ylabel("accuracy rate")
	plt.legend()
	plt.grid()
	plt.show()


if __name__ == "__main__":


	#############################################################################
	#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
	#############################################################################
	d_list = [100, 200, 400, 800, 2000, 3000, 5000]
	tp = exprmt_d(100, d_list, 0.1)
	plot_accuracy(tp, d_list)


