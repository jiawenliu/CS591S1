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
	data = []
	for _ in range(n):
		x = np.random.normal(0.7, 0.01)
		data.append(0.0 if x < 0.0 else 0.99 if x > 1.0 else x)

	return np.array(data)

def gen_datasizes(r, step):
	return [i*step for i in range(r[0]/step,r[1]/step + 1)]

#############################################################################
#SETTING UP THE GRANULARITY OF T
#############################################################################

def gen_t():
	return [0.1 * i for i in range(10)]


#############################################################################
#SIMPLE HISTOGRAM ALGORITHM
#############################################################################
def Simple_Histogram(x, a, eps):
	Y = [0]
	n = len(x)
	for j in range(1, int(1/a) + 1):
		I0, I1 = (j - 1) * a, j * a
		count = 0
		for xi in x:
			count += 1 if (xi < I1 and xi >= I0) else 0.0
		Y.append(count/n + np.random.laplace(0.0, 2/(eps * n)))
	return Y

#############################################################################
#SIMPLE HISTOGRAM CDF ALGORITHM
#############################################################################
def Simple_Histogram_CDF(x, a, eps):
	Y = Simple_Histogram(x, a, eps)
	n = len(x)	
	def eCDF(t):
		return sum([Y[j] for j in range(1, int(math.floor(t/a)))]) + (t/a - math.floor(t/a)) * Y[int(math.floor(t/a)) + 1]
	
	ecdf = np.array([eCDF(t) for t in gen_t()])

	return ecdf



#############################################################################
#TREE HISTOGRAM CDF ALGORITHM
#############################################################################
def Tree_Histogram_CDF(x, l, eps):
	va = [(1.0/2) ** i for i in range(1, l + 1)]
	vecdf = [Simple_Histogram_CDF(x, a, eps) for a in va]

	ecdf = sum(vecdf)/l

	return ecdf

#############################################################################
# CDF FUNCTION
#############################################################################
def CDF(x):
	n = len(x)
	def CDFt(t):
		count = 0.0
		for xi in x:
			if xi < t: count += 1.0
		return count/n
	return np.array([CDFt(t) for t in gen_t()])


#############################################################################
# ERROR COMPUTING
def error(cdf, ecdf):
	return max(abs(cdf - ecdf))

#############################################################################
# TEST SIMPLE OR TREE CDF FUNCTION
def expermt(eps, n, a):
	e = 0.0
	for _ in range(20):
		x = gen_dataset(n)
		ecdf = Simple_Histogram_CDF(x, a, eps)
		cdf = CDF(x)
		e += error(cdf, ecdf)
	print a, e
	return e/20.0

def expermt_va(eps, n, va):
	return [expermt(eps, n, a) for a in va]

def plot_accuracy(ys, ns):
	plt.figure()
	plt.plot(ns, ys, "ro-", label = "Error")
	plt.xlabel(r'$l$')
	plt.ylabel("Error")
	plt.title(r"Simple Histogram CDF with n = $10^5$")
	plt.legend()
	plt.grid()
	plt.show()


if __name__ == "__main__":
	#############################################################################
	#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
	#############################################################################
	eps = 0.5
	n = 100000
	va = [(1.0/2) ** i for i in range(12, 16)]
	vl = range(1, 16)
	plot_accuracy(expermt_va(eps, n, va), range(12, 16 ))


