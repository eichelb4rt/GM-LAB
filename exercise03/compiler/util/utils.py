#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

Helper functions for plotting, file-io
'''
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path

from compiler.util.org import DIR_NX

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def ensure_dir_exists(dir_):
	print ("Deprecated: use ensure_path_exists.")
	# TODO: make independent of filesystem
	subdirs = dir_.split('/')

	current_dir = ''
	for i in range(len(subdirs)):
		current_dir += subdirs[i]
		if not os.path.exists(current_dir):
			os.mkdir(current_dir)
		current_dir += '/'

def ensure_path_exists(path):
	current_path = Path('.')
	for p in path.parts:
		current_path = current_path / p
		if not current_path.exists():
			current_path.mkdir()

def check_file_exists(filepath):
	current_path = Path('.')
	for p in filepath.parts:
		current_path = current_path / p
		if not current_path.exists():
			return False
			#raise IOError("Does not exist: "+str(current_path))
	return True

def print_progress(part, total, front_string="Progress:", end_string=""):
	if not total == 0:
		print (front_string+" "+str("%6.2f" % ((float(part)/(float(total)/100)))) + "% "+end_string, end='\r')
		if part >= total:
			print ()
		sys.stdout.flush()

def plot_nx_graph(G, filename, tree=False, labels=True, colored_nodes=[]):
	if tree:
		pos = graphviz_layout(G, prog='dot')
		nx.draw(G, pos, with_labels=labels)
		if len(colored_nodes) > 0:
			nx.draw_networkx_nodes(G, pos, nodelist=colored_nodes, node_color="tab:red")
	else:
		nx.draw(G, with_labels=labels)
		if len(colored_nodes) > 0:
			nx.draw_networkx_nodes(G, nodelist=colored_nodes, node_color="tab:red")
	ensure_path_exists(DIR_NX)
	full_filename = filename+".png"
	plt.savefig(DIR_NX / full_filename)
	plt.close()