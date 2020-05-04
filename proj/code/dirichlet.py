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


def gen_betaln (alphas):
	numerator=0.0
	for alpha in alphas:
		numerator = numerator + gammaln(alpha)
	return(numerator/math.log(float(sum(alphas))))


def opt_hellinger(dirichlet1, dirichlet2):
	alphas = deepcopy(dirichlet1._alphas)
	betas = deepcopy(dirichlet2._alphas)
	z=gen_betaln(numpy.divide(numpy.sum([alphas, betas], axis=0), 2.0))-0.5*(gen_betaln(alphas) + gen_betaln(betas))
	return (math.sqrt(1-math.exp(z)))



def L1_Nrom(A, B):
	return numpy.sum(abs(numpy.array(A._alphas) - numpy.array(B._alphas)))


class dirichlet(object):
	def __init__(self, alphas):
		self._alphas = alphas
		self._size = len(alphas)

	def __sub__(self, other):
		return opt_hellinger(self, other)
		# return Optimized_Hellinger_Distance_Dir(self, other)

	def _minus(self,other):
		self._alphas = list(numpy.array(self._alphas) - numpy.array(other._alphas))
		return self

	def _pointwise_sub(self, other):
		return dirichlet(list(numpy.array(self._alphas) - numpy.array(other._alphas)))

	def __add__(self, other):
		return dirichlet(list(numpy.array(self._alphas) + numpy.array(other._alphas)))

	def show(self):
		print "Dirichlet("+str(self._alphas) + ")"

	def _hellinger_sensitivity(self):
		LS = 0.0
		r = deepcopy(self)
		temp = deepcopy(r._alphas)
		for i in range(0, self._size-1):
			temp[i] += 1
			# print temp
			for j in range(i + 1, self._size):
				temp[j] -= 1
				# print temp
				if temp[j]<=0:
					temp[j] += 1
					continue
				LS = max(LS, abs(dirichlet(temp) - self))
				# print r._alphas,self._alphas,temp,(r-self),((temp) - self)
				temp[j] += 1
			temp[i] -= 1
		for i in range(0, self._size-1):
			temp[i] -= 1
			if temp[i]<=0:
					temp[i] += 1
					continue
			# print temp
			for j in range(i + 1, self._size):
				temp[j] += 1
				# print temp
				LS = max(LS, abs(dirichlet(temp) - self))
				# print r._alphas,self._alphas,temp,(r-self),((temp) - self)
				temp[j] -= 1
			temp[i] += 1		
		return LS


	def _score_sensitivity(self, r):
		LS = 0.0
		temp = deepcopy(self._alphas)
		for i in range(0, self._size):
			temp[i] += 1
			for j in range(i + 1, self._size):
				temp[j] -= 1
				LS = max(LS, abs(-(r - self) + (r - (temp))))
				temp[j] += 1
			temp[i] -= 1
		return LS
