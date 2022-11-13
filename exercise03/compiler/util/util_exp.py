#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

utility functions for experiments
'''
import numpy as np
import random as rd

import os
import psutil
import time
import pickle
import csv

from func_timeout import func_timeout, FunctionTimedOut

from DataStructure.Models import Model
from util.LoggingManager import LoggingManager as LM
from util import utils


# param_min, param_max: bounds of random parameters in generated q
PARAM_MIN = 0.1
PARAM_MAX = 0.9
PARAM_RANGE = PARAM_MAX - PARAM_MIN

### generating artificial data
def construct_random_query(experiment_type, pct_marg, pct_cond, n):
	'''
	experiment_type is "MPE" or "POE"
	if "MPE":
		- marginalizing is not allowed
		- query is non-empty
		- input_evidence is empty
	else:
		- query is empty
		- input_evidence is non-empty

	pct_marg and pct_cond describe percentages of variables to marginalize and condition
	n is total number of variables
	'''
	if experiment_type == "MPE":
		# ensure that we don't end up with a mixture of gaussians after marginalizing:
		pct_marg = 0

	# compute sizes of sets
	num_marg = int(n*pct_marg//100)
	num_cond = int(n*pct_cond//100)
	num_input = n-num_cond-num_marg
	
	# indices of conditions:
	cond = rd.sample([i for i in range(n)], num_cond)
	# evidence:
	evidence = [rd.randint(0,1) for _ in range(num_cond)] 
	
	# indicies of marginalized variables:
	marg = rd.sample([k for k in range(n) if not k in cond], num_marg)
	# indices of query-variables (case MPE)
	query = rd.sample([i for i in range(n) if not i in cond], n-num_cond-num_marg)
	# input variables (case POE)
	input_evidence = [rd.randint(0,1) for _ in range(num_input)]

	return query, input_evidence, marg, cond, evidence

def construct_random_q_from_graph(G, d=2, random_params=True, nonzero_potentials=False):
	# G is a networkx-graph
	# construct an interaction-matrix q that
	# -> has random interaction parameters for every edge in G
	# -> has zero interaction for every independent pair of nodes in G
	n = len(G.nodes())

	q = np.zeros((n,n,d,d))
	for e in G.edges():
		v_0 = min(e[0],e[1])
		v_1 = max(e[0],e[1])
		if not random_params:
			q[v_0, v_1] = np.ones((d,d))	
		else:
			# set q[i,j] to random dxd-interaction
			q[v_0, v_1] = np.random.random_sample((d,d))*PARAM_RANGE+PARAM_MIN

	if nonzero_potentials:
		for i in range(n):
			q[i,i] = np.eye((d,d))*np.random.rand()*PARAM_RANGE+PARAM_MIN
	return q

def construct_random_q(n, d, tau, nonzero_potentials=False):
	'''
	n: number of variables
	d: number of states per variable
	tau: percentage of interactions in q
	nonzero_potentials: if False, then the main diagonal of q is zero
	'''
	if tau < 0 or tau > 100:
		tau = 100

	q = np.zeros((n,n,d,d))
	max_interactions = int((n-1)*n/2)
	num_interactions = int(max_interactions * tau/100)

	all_var_pairs = [[i,j] for i in range(n-1) for j in range(i+1,n)]
	var_pairs_choice = rd.sample(all_var_pairs, num_interactions)

	for var_pair in var_pairs_choice:
		q[var_pair[0],var_pair[1]] = np.random.random_sample((d,d))*PARAM_RANGE+PARAM_MIN

	if nonzero_potentials:
		for i in range(n):
			for j in range(d):
				q[i,i,j,j] = np.random.rand()*PARAM_RANGE+PARAM_MIN
	return q

def construct_query_params(n, pct_m, pct_c):
	'''
	Construct a random query
	'''
	num_marg = int(n*pct_m // 100)
	num_cond = int(n*pct_c // 100)
	num_input_poe = n - num_marg - num_cond
	num_query_mpe = n - num_cond

	cond = rd.sample([i for i in range(n)], num_cond)
	evidence = [0 for _ in range(num_cond)]
	marg = rd.sample([i for i in range(n) if i not in cond], num_marg)

	input_poe = [0 for _ in range(num_input_poe)]
	query_mpe = [i for i in range(n) if not i in cond]

	return input_poe, query_mpe, marg, cond, evidence

### HELPER FUNCTIONS
def get_filename(exp_type, n, d, tau, timeout_compile, timeout_query):
	return "data_exp-"+exp_type+"_n"+str(n)+"_d"+str(d)+"_ial"+str(tau)+"_t"+str(timeout_query)+"_tc"+str(timeout_compile)+".json"

@LM.logfunc
def safe_run_with_timeout(function, timeout, args):
	'''
	Run a function with timeout.
	Specified on compiling an LH model: common exceptions that might happen during compilation are catched
	and log-messages are specified on compilation.

	Return:
		success: boolean flag that is True if function finished without errors within timelimit
		return_: the return values of the function
	'''
	success = 0
	return_ = None
	try:
		return_ = func_timeout(timeout, function, args=args)
		success = 1
		LM.log("compiling successfull.")
	except FunctionTimedOut:
		LM.log("!! TIMEOUT during _compile_tree !!")
		LM.log(" Killing ALL running instances of C2D compiler and child processes...")
		current_process = psutil.Process()
		children = current_process.children(recursive=True)
		for child in children:
			LM.log('  Kill child {}'.format(child.pid))
			child.kill()

		LM.log(" Deleting all remaining temporary files from killed C2D-compiler")
		for file in os.listdir():
			if ".tmp" in file:
				LM.log("  Delete file {}".format(file))
				os.remove(file)
		success = -1
	except FileNotFoundError:
		LM.log("!! Out of memory during _compile_tree !!")
		LM.log(" Killing ALL running instances of C2D compiler and child processes...")
		current_process = psutil.Process()
		children = current_process.children(recursive=True)
		for child in children:
			LM.log('  Kill child {}'.format(child.pid))
			child.kill()

		LM.log(" Deleting all remaining temporary files from killed C2D-compiler")
		for file in os.listdir():
			if ".tmp" in file:
				LM.log("  Delete file {}".format(file))
				os.remove(file)
		success = -2
	except Exception as e:
		LM.log("!! Unknown Error during compile (in c2d-compiler?) !!")
		LM.log(" Killing ALL running instances of C2D compiler and child processes...")
		print (e)
		current_process = psutil.Process()
		children = current_process.children(recursive=True)
		for child in children:
			LM.log('  Kill child {}'.format(child.pid))
			child.kill()

		LM.log(" Deleting all remaining temporary files from killed C2D-compiler")
		for file in os.listdir():
			if ".tmp" in file:
				LM.log("  Delete file {}".format(file))
				os.remove(file)
		success = -3
		raise
	return success, return_

@LM.logfunc
def get_model(n, d, tau, timeout_compile, compile_ac=True, normalize=True):
	'''
	Generates a random LH model: generates parameters and compiles the model

	Args:
		n : number of discrete variables
		d : dimension of discrete variables
		m : number of continuous varaibles
		tau : density of model (percentage of all possible interactions present in model)

		timeout_compile : timeout in seconds for model construction (including compiling the ET)
		compile_ac : if False, no AC is compiled.
		normalize : if False, model is not normalized

	Return:
		success: a boolean flag
		lh: the compiles LH-model
		t_compile : the compile time
	'''
	q = construct_random_q(n,d, tau)

	num_nodes = -1
	num_components = -1
	components = []
	t_compile = -1
	model = None

	success, return_ = safe_run_with_timeout(compile_model, timeout_compile, (n, d, q, compile_ac, normalize))
	if success > 0:
		lh, t_compile = return_

	return success, lh, t_compile

@LM.logfunc
def get_model_with_q(n, d, q, timeout_compile, compile_ac=True, normalize=True):
	'''
	Generates a random LH model: generates parameters and compiles the model

	Args:
		n : number of discrete variables
		d : dimension of discrete variables
		q : model interactions

		timeout_compile : timeout in seconds for model construction (including compiling the ET)
		compile_ac : if False, no AC is compiled.
		normalize : if False, model is not normalized

	Return:
		success: a boolean flag
		lh: the compiles LH-model
		t_compile : the compile time
	'''

	num_nodes = -1
	num_components = -1
	components = []
	t_compile = -1
	model = None

	success, return_ = safe_run_with_timeout(compile_model, timeout_compile, (n, d, q, compile_ac, normalize))
	if success > 0:
		lh, t_compile = return_

	return success, lh, t_compile

@LM.logfunc
def compile_model(n, d, q, compile_ac=True, normalize=True):
	'''
	Constructs (& compiles) a LH-model from parameters.
	'''
	LM.log("q:")
	LM.log(str(q))
	
	if not normalize:
		const = 1
	else:
		const = None
	time_compile_start = time.time()
	model = Model.Model(n, d, q, const=const, compile_AC=compile_ac)
	time_compile = time.time() - time_compile_start
	return model, time_compile

def store_model(model, basedir, iteration=None):
	'''
	Write a lh-model to a file in human-readable form.
	Might NOT include the full parameter tensors if they are big.

	Args:
		model: the lh-model
		basedir: a Path
		iteration: an optional int to enumerate the model
	'''
	modeldir = basedir / "models"
	if iteration == None:
		filename = "model.txt"
	else:
		filename = "model_"+str(iteration)+".txt"
	model.store_hr(filename, modeldir)

def pickle_model(model, basedir, filename="model"):
	'''
	Store lh-model in pickle
		model: the model
		basedir: a Path
		filename: name without .pickle
	'''
	modeldir = basedir / "models"
	utils.ensure_path_exists(modeldir)
	with open(str(modeldir)+"/"+filename+".pickle", 'wb') as handle:
		pickle.dump(model, handle)

def load_pickled_model(modeldir, filename="model"):
	'''
	Load lh-model from pickle
		dir: a Path
		filename: name without .pickle
	'''
	with open(str(modeldir)+"/"+filename+".pickle", 'rb') as handle:
		lh_loaded = pickle.load(handle)
		return lh_loaded

def analyze_model(model, visualize_gm=True, filename="gm", threshold=1e-6):
	'''
	Print statistics of a LH model for analyzing.

	Args:
		lh: a LH-model
		viszalize_gm: boolean. if True, the graphical model is plotted
		filename: a sring that specifies a filename for the plotted GM.
		threshold: a lower bound on edge intensities used for plotting the GM.
	'''
	nodes, edges, components = model.get_graphical_model()
	components = model.get_graphical_model(consider_mixed=False, th=threshold)
	largest_component = max([len(ci) for ci in components])

	# print the statistics:
	print ("num nodes",len(nodes))
	print ("num edges",len(edges))
	print ("num components",len(components))
	print ("size of largest component:",largest_component)

	lh.m = 0
	lh.nu = [[[]]]

	if visualize_gm:
		model.visualize_graphical_model(True, filename=filename, th=threshold)
	
def construct_gridlike_graph(s=[4,5]):
	'''
	For the resulting graph to contain no cliques,
	the elements of s have to satisfy:
	- larger than 3,
	- pairwise share no common divisors
	size of constucted graph is product of all elements from s.
	'''
	n = np.prod(s)
	G = nx.Graph()
	G.add_nodes_from([_ for _ in range(n)])
	edges = []
	for i in range(n):
		for stepwidth in s:
			edges.append((i, (i+stepwidth)%n))
	G.add_edges_from(edges)
	return G
