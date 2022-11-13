#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

Class for constructing the mean-parameterization (p, mu, Sigma) from a CG distribution given in canonical parameterization.
'''
import numpy as np
import math
import itertools

from InferenceEngine.DataStructure.LHmodels import LH

def lh_to_meanparams(lh_model):
	'''
	Construct the mean parameterization of a CG distribution given as CG_vec object.

	constructs p, mu, Sigma as follows:
		p is a dictionary of floats, that has tuples as keys.
		In particular, for every possible assignment of values x1, x2, ..., xn 
		to all discrete variabels X1, ..., Xn; p([x1, ...,xn]) contains the marginal probability of (x1, ..., xn)
		
		mu is a dictionary of lists of floats, that has tuples as keys.
		Similar to p, mu contains an element for each possible assignment of the discrete variables.
		The elements of mu are the mean values of the continouos variables conditioned on the specific given
		assignment of the discrete variables.

		Sigma is a matrix,
		simply the inverse of the matrix Lambda that defines the pairwise interactions between continouos variables.
	'''
	p = {}
	mu = {}

	# construct Sigma:
	if not lh_model.Lambda.size == 0:
		Sigma = np.linalg.inv(lh_model.Lambda)
	else:
		Sigma = np.asarray([[]])

	# construct mu:
	Omega_X = itertools.product(*[range(lh_model.N[i]) for i in range(lh_model.n)])
	for x in Omega_X:
		nu_x = [0 for _ in range(lh_model.m)]
		for j in range(lh_model.m):
			for i in range(lh_model.n):
				nu_x[j] += lh_model.nu[i, x[i], j]

		if len(lh_model.alpha) > 0:
			for j in range(lh_model.m):
				nu_x[j] += lh_model.alpha[j]
		mu[x] = np.matmul(Sigma, nu_x)

	# construct p:
	Omega_X = itertools.product(*[range(lh_model.N[i]) for i in range(lh_model.n)])
	if not lh_model.m == 0:
		#sqrt_of_det_L = math.sqrt(np.linalg.det(lh_model.Lambda))

		for x in Omega_X:
			log_q_x = 0
			nu_x = np.zeros(lh_model.m)
			for i in range(lh_model.n):
				log_q_x += 0.5*lh_model.q[i, i, x[i], x[i]]
				for j in range(i+1,lh_model.n):
					log_q_x += lh_model.q[i, j, x[i], x[j]]

				for j in range(lh_model.m):
					nu_x[j] += lh_model.nu[i, x[i], j]
					if len(lh_model.alpha) > 0:
						nu_x[j] += lh_model.alpha[j]

			muT_L_mu = np.matmul(np.matmul(np.transpose(nu_x),Sigma),nu_x)
			p[x] = np.exp(log_q_x + 0.5*muT_L_mu)
	else:
		for x in Omega_X:
			log_q_x = 0
			for i in range(lh_model.n):
				log_q_x += 0.5*lh_model.q[i, i, x[i], x[i]]
				for j in range(i+1,lh_model.n):
					log_q_x += lh_model.q[i, j, x[i], x[j]]
			p[x] = np.exp(log_q_x)

	p_total = np.sum([p[x] for x in p.keys()])
	p = {x: p[x]/p_total for x in p}

	return p, mu, Sigma

def sample_from_cg(p, mu, Sigma):
	'''
	A simple function to sample from a CG distribution given in mean parameterization.
	'''
	x = np.random.choice(len(p), 1, p)[0]
	_mu = [mu[x] for x in mu.keys()]
	mu_x = _mu[x]		
	y = np.random.multivariate_normal(mu_x, Sigma)

	return x,y