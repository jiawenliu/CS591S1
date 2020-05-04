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
from dirichlet import dirichlet
from scipy.special import gammaln


def Hamming_Distance(c1, c2):
	temp = [abs(a - b) for a,b in zip(c1,c2)]
	return sum(temp)/2.0


class BayesInferwithDirPrior(object):
	def __init__(self, prior, sample_size, epsilon, delta = 0.0000000001, gamma=1.0):
		self._prior = prior
		self._sample_size = sample_size
		self._epsilon = epsilon
		self._delta = delta
		self._gamma = gamma
		self._bias = numpy.random.dirichlet(self._prior._alphas)
		self._observation = numpy.random.multinomial(1, self._bias, self._sample_size)
		self._observation_counts = numpy.sum(self._observation, 0)
		self._posterior = dirichlet(self._observation_counts) + self._prior
		self._laplaced_posterior = self._posterior
		self._laplaced_geo_posterior = self._posterior
		self._keys = []
		self._accuracy = {}
		self._accuracy_mean = {}

	def _set_gamma(self, gamma):
		self._gamma = gamma
		
	def _set_bias(self, bias):
		self._bias = bias
		self._update_observation()

	def _set_observation(self,observation):
		self._observation_counts = observation
		self._posterior = dirichlet(observation) + self._prior



	###################################################################################################################################
	#####SETTING UP THE BASELINE LAPLACE MECHANISM
	###################################################################################################################################	


	def _set_up_naive_lap_mech(self):
		self._keys.append("Laplace Noise")
		self._accuracy["Laplace Noise"]=[]
		self._accuracy_mean["Laplace Noise"]=[]

	def _set_up_geo_lap_mech(self):
		self._keys.append("Geometric Noise")
		self._accuracy["Geometric Noise"]=[]
		self._accuracy_mean["Geometric Noise"]=[]


	def _laplace_mechanism_naive(self):
		noised = [i + math.floor(numpy.random.laplace(0, 2.0/self._epsilon)) for i in self._observation_counts]
		noised = [self._sample_size if i > self._sample_size else 0.0 if i < 0.0 else i for i in noised]

		self._laplaced_posterior = dirichlet(noised) + self._prior

	def _laplace_mechanism_geo(self):
		noised = [i + numpy.random.geometric(numpy.exp(-self._epsilon)) for i in self._observation_counts]
		noised = [self._sample_size if i > self._sample_size else 0.0 if i < 0.0 else i for i in noised]

		self._laplaced_geo_posterior = dirichlet(noised) + self._prior


########################################################################################################################################
######EXPERMENTS FRO N TIMES, I.E., CALL THE SAMPLING FUNCTION OF EACH MECHANISMS FOR N TIMES
########################################################################################################################################	

	def _experiments(self, times):
		self._set_up_naive_lap_mech()
		self._set_up_geo_lap_mech()

		for i in range(times):
			#############################################################################
			self._laplace_mechanism_naive()
			print  "Lap", self._laplaced_posterior._alphas
			self._accuracy[self._keys[0]].append(self._posterior - self._laplaced_posterior)

			self._laplace_mechanism_geo()
			print  "Geo", self._laplaced_geo_posterior._alphas
			self._accuracy[self._keys[1]].append(self._posterior - self._laplaced_geo_posterior)


			
		for key,item in self._accuracy.items():
			self._accuracy_mean[key] = numpy.mean(item)


########################################################################################################################################
######PRINT FUNCTION TO SHOW THE PRARMETERS OF THE CLASS
########################################################################################################################################	

	def _get_bias(self):
		return self._bias

	def _get_observation(self):
		return self._observation

	def _get_posterior(self):
		return self._posterior


	def _show_bias(self):
		print "The bias generated from the prior distribution is: " + str(self._bias)

	def _show_laplaced(self):
		print "The posterior distribution under Laplace mechanism is: "
		self._laplaced_posterior.show()


	def _show_observation(self):
		print "The observed data set is: "
		print self._observation
		print "The observed counting data is: "
		print self._observation_counts

	def _show_prior(self):
		print "The prior distribution is: "
		self._prior.show()

	def _show_all(self):
		self._show_prior()
		self._show_bias()
		self._show_observation()
		self._show_laplaced()
		self._show_exponential()
