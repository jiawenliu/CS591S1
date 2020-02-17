import numpy as np
import matplotlib.pyplot as plt
import random

#GENERATING DATA SIZE AND CONRRESPONDING PARAMETER
def gen_data(d, n,  M, sigma):
	unnormx = [random.choice(M) + np.array([random.gauss(0, sigma) for _ in range(d)]) for i in range(n)]
	return np.array([ x / sum(abs(x)) if sum(abs(x)) > 1 else x for x in unnormx]) 

def gen_point(d):
	p = np.array([random.uniform(-1, 1) for _ in range(d)])
	while sum(abs(p)) > 1.0:
		p = np.array([random.uniform(-1, 1) for _ in range(d)])
	return np.array(p)

#PURE K_MEAN ALGORITHM
def k_clustering(data, T, eps, k):
	ctrs = np.array([gen_point(len(data[0])) for _ in range(k)]) #initialize the centers
	ctrs_record = []
	for _ in range(T):
		distance, S, used = [[sum(abs(row - c)) for c in ctrs] for row in data], [[] for _ in range(k)], [False]*len(data)
		for j in range(k):
			for i in range(len(data)):
				if distance[i][j] <= min(distance[i]) and not used[i]:
					used[i] = True
					S[j].append(data[i])
		ctrs = [sum(S[i])/ len(S[i]) if S[i] != [] else gen_point(len(data[0])) for i in range(k)]
		ctrs_record.append(np.array(ctrs))
	return np.array(ctrs), np.array(ctrs_record)

#PRIVATE VERSION OF K_MEAN ALGORITHM
def k_clustering_private(data, T, eps, k):
	ctrs = np.array([gen_point(len(data[0])) for _ in range(k)]) #initialize the centers
	eps, ctrs_record = eps/(2.0 * T), []
	for _ in range(T):
		distance, S, used = [[sum(abs(row - c)) for c in ctrs] for row in data], [[] for _ in range(k)], [False]*len(data)
		for j in range(k):
			for i in range(len(data)):
				if distance[i][j] <= min(distance[i]) and not used[i]:
					S[j].append(data[i])
					used[i] = True # prevent two data point added to two different sets when they have the same distance
		ctrs = [(sum(S[i]) + np.random.laplace(0.0, eps))/ (len(S[i]) + np.random.laplace(0.0, 2*eps)) if len(S[i]) > 5 else gen_point(len(data[0])) for i in range(k)]
		ctrs_record.append(np.array(ctrs))
	return np.array(ctrs), np.array(ctrs_record)


def plot_accuracy(data, centers, M):
	def obx (data):
		return list(data.transpose()[0])
	def oby (data):
		return list(data.transpose()[1])
	def ithcx(data, i):
		return data.transpose()[0][i]
	def ithcy(data, i):
		return data.transpose()[1][i]
	plt.figure()
	plt.plot(obx(data), oby(data), "o", color="orchid", label="data")
	for i in range(len(M)):
		plt.plot((ithcx(centers, i)), (ithcy(centers, i)), "*--", label="centers" + str(i))
	plt.plot(obx(M), oby(M), "g^", label="M")
	plt.legend()
	plt.grid()
	plt.show()




if __name__ == "__main__":

	#SETTING UP THE PARAMETERS WHEN DOING GROUPS EXPERIMENTS
	M = np.array([[0, 0.5], [0.2, -0.2], [-0.2, -0.2]])
	data = gen_data(2, 2000, M, 0.01)
	#EXPERIMENTS WIEH PRIVATE K_MEANS
	centers, records = k_clustering_private(data, 10, 0.1, 3)
	plot_accuracy(data, records, M)
	#EXPERIMENTS WIEH PURE K_MEANS
	centers, records = k_clustering(data, 10, 0.1, 3)
	plot_accuracy(data, records, M)




	

