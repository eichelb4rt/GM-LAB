#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

Classes for arithmetic circuits
'''

import math
from collections import deque
from enum import Enum
from pathlib import Path
import itertools as it
import copy as cp

import numpy as np

import networkx as nx

from compiler.ArithmeticCircuit import Expressions as expr
from compiler.ArithmeticCircuit import Node
from compiler.ArithmeticCircuit.Node import Node_leaf_param

from compiler.util.LoggingManager import LoggingManager as LM
from compiler.util import utils
from compiler.util.org import DIR_GRAPH

class State():
	_ids = it.count(0)

	def __repr__(self):
		return f"{self.id}: {self.state}"

	def __str__(self):
		return self.__repr__()

	def __init__(self):
		self.id = next(self._ids)
		self.state = {}

	def add_entry(self, variable, value):
		assert not variable in self.state, "Variable should not be in the state"
		self.state[variable] = value

	def clone(self):
		s = State()
		s.state = cp.deepcopy(self.state)
		return s

class ArithmeticCircuit:
	'''
	Class for an arithmetic circuit

	Args:
		root (Node):
			the root node of the arithmetic circuit. The full tree is only stored via the children of the root.
	'''
	@LM.logfunc
	def __init__(self, root=None):
		self.root = root
		self.num_nodes = -1

	def __str__(self):
		str_ = ""

		# traverse tree in pre-order:
		nodestack = [self.root]
		while len(nodestack) > 0:
			current_node = nodestack[-1]
			nodestack.pop(-1)
			cn_str = current_node.get_description()
			if len(cn_str) > 0:
				str_ += cn_str+"\n"
			for c in current_node.children:
				nodestack.append(c)
		return str_

	def set_root(self, node):
		self.root = node

	def copy(self):
		'''
		returns a deep copy of this arithmetic circuit
		'''
		root_copy = self.root.copy(new_parent=None)
		current_node_copy = root_copy

		node_stack = [[self.root,0]]
		while len(node_stack)>0:
			#print ([n[0].id for n in node_stack])
			[current_node, current_child_id] = node_stack[-1]
			if current_child_id >= current_node.get_num_children():
				# in copied tree: go back to parent:
				current_node_copy = current_node_copy.parent

				# remove node from stack
				node_stack.pop(-1)
			else:
				# copy child:
				child_copy = current_node.children[current_child_id].copy(current_node_copy)
				# set pointer in copied tree to copied child:
				current_node_copy = child_copy
				# increase child index:
				node_stack[-1][1] += 1
				# push child onto stack:
				node_stack.append([current_node.children[current_child_id], 0])
		return ArithmeticCircuit(root=root_copy)

	@LM.set_log_1
	@LM.logfunc_no_sublog
	def simplify(self, start_node=None):
		'''
		simplifies the ET in the following cases:
			1) removes the following nodes:
				1a) internal nodes without children
				1b) parameter-node-children of addition-nodes whose parameter is 0
				1c) parameter-node-children of multiplication-nodes whose parameter is 1
				1d) internal addition- or multiplication-nodes that have only one child 
				 (these get replaced with their child)
			2) pre-computes fixed subtrees:
				2a) merges internal nodes that do not depend on variables into parameter-leaf-nodes
				2b) merges sets of parameter nodes that are children of the same internal node
		Also make the computation more numerically stable:
			3) if a multiplication node has multiple children that are exp-nodes, these
				exp-nodes are merged into a single exp-node with a sum-node-child

		if cont_values is specified, then
			4) the input nodes for continuous variables are replaced with parameter nodes that have
				the corresponding value from cont_values

		This algorithm also constructs the scopes and substates of the nodes.
		
		Args:
			cont_values:
				if not None, this is required to contain one float for each continuous variable.
				In this case, the continuous input nodes are replaced with parameter nodes with the
				corresponding value.
			start_node:
				if None, simplify starts at the root and simplifies the complete tree.
				Otherwise, simplify starts at the specified node an only simplifies the sub-tree.
		'''
		node_stack = self._construct_post_order(start_node=start_node)

		for node in node_stack:
			node.simplify()
			node.set_scope()

	def simplify_parameters(self, node=None, current_parameter=1):
		'''
		Make the arithmetic circuit smaller by pushing parameter nodes down.
		'''
		if node is None:
			node = self.root
		if isinstance(node, Node.Node_internal):
			if node.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
				for c in node.children:
					self.simplify_parameters(c, current_parameter)
			else:
				internal_children = []
				parameter_children = []
				for c in node.children:
					if isinstance(c, Node.Node_internal):
						internal_children.append(c)
					elif isinstance(c, Node.Node_leaf_param):
						parameter_children.append(c)
						current_parameter *= c.value
				for c in parameter_children:
					node.children.remove(c)
				if len(internal_children) == 0:
					# leaf node reached. Add parameter node:
					Node.Node_leaf_param(parent=node, theta=current_parameter)
				elif len(internal_children) == 1:
					# only one subtree: continue to collect parameters
					for c in internal_children:
						self.simplify_parameters(c, current_parameter) 
				else:
					# multiple subtrees:
					# Add parameter node with parameters collected so far and
					# start new parameter collection for subtrees:
					if current_parameter != 1:
						Node.Node_leaf_param(parent=node, theta=current_parameter)
					for c in internal_children:
						self.simplify_parameters(c)

	def simplify_distributive(self, start_node=None):
		'''
		Make the arithmetic circuit smaller by applying the distributive law:
		pushing indicator nodes as high as possible
		'''
		def get_singular_domains(start_node, known_domains={}):
			#print ("get get_singular_domains:", start_node)
			node_stack = self._construct_post_order(start_node=start_node, consider_leaves=True)
			for node in node_stack:
				if isinstance(node, Node.Node_leaf_input):
					known_domains[node.id] = {node.var_id : node.cat_value}
				elif isinstance(node, Node.Node_leaf_param):
					known_domains[node.id] = {}
				elif isinstance(node, Node.Node_internal):
					known_domains[node.id] = {}
					if node.expr_type == expr.ExpressionNodeType.ETYPE_MUL:
						# merge domains (note that domains of children are disjoint by decomposability):
						for c in node.children:
							for var_id in known_domains[c.id]:
								known_domains[node.id][var_id] = known_domains[c.id][var_id]
					if node.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
						for c in node.children:
							for var_id in known_domains[c.id]:
								if var_id in known_domains[node.id]:
									if known_domains[c.id][var_id] != known_domains[node.id][var_id]:
										# remove var_id for variable with non-singular domain:
										known_domains[node.id].pop(var_id, None)
								else:
									# add new var_id:
									known_domains[node.id][var_id] = known_domains[c.id][var_id]
			return known_domains

		def move_indicator_up(parent, relevant_nodes, var_id, value, known_domains):
			#print ("move indicator up at", parent.id, var_id)
			relevant_children = [c for c in parent.children if c in relevant_nodes]
			#print ("move up indicator X_"+str(var_id)+"="+str(value))
			# add product node as new child of current node:
			prod_node = Node.Node_internal(expr_type=expr.ExpressionNodeType.ETYPE_MUL, parent=parent)
			# add indicator variable as child of product node:
			indicator = Node.Node_leaf_input(parent=prod_node, var_id=var_id, cat_value=value)
			# add sum node as child of product node:
			sum_node = Node.Node_internal(expr_type=expr.ExpressionNodeType.ETYPE_ADD, parent=prod_node)
			# reorganize all children_with_var to be children of sum_node:
			for c in relevant_children:
				c.set_parent(new_parent=sum_node)
			# delete all indicator variables for var_id in subtree of sum_node:
			nodes = self._construct_post_order(start_node=sum_node, consider_leaves=True)
			for n in nodes:
				if n.id in known_domains and var_id in known_domains[n.id]:
					known_domains[n.id].pop(var_id, None)
				if isinstance(n, Node.Node_leaf_input):
					if n.var_id == var_id:
						n.parent.children.remove(n)
			return sum_node, known_domains

		def process_sum_node(node, node_domains):
			#print ("process sum node:", node.id)
			# check if distributive law can be applied:
			children_with_equal_singular_domains = {}
			for c_i in range(len(node.children)):
				for var_id in node_domains[node.children[c_i].id]:
					val_i = node_domains[node.children[c_i].id][var_id]
					for c_j in range(c_i, len(node.children)):
						if var_id in node_domains[node.children[c_j].id]:
							val_j = node_domains[node.children[c_j].id][var_id]
							if val_i == val_j:
								if var_id not in children_with_equal_singular_domains:
									children_with_equal_singular_domains[var_id] = {}
								if val_i not in children_with_equal_singular_domains[var_id]:
									children_with_equal_singular_domains[var_id][val_i] = [node.children[c_i], node.children[c_j]]
								else:
									children_with_equal_singular_domains[var_id][val_i].append(node.children[c_j])
			for var_id in children_with_equal_singular_domains:
				for value in children_with_equal_singular_domains[var_id]:
					new_sum, node_domains = move_indicator_up(node, children_with_equal_singular_domains[var_id][value], var_id, value, node_domains)
					process_sum_node(new_sum, node_domains)

		node_domains = {}
		node_stack = self._construct_post_order(start_node=start_node, consider_leaves=True)
		for node in node_stack:
			node_domains = get_singular_domains(node, node_domains)
			if isinstance(node, Node.Node_internal) and node.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
				process_sum_node(node, node_domains)
				self.simplify(start_node=node)

	def get_num_nodes(self, recompute=True):
		# todo: fix bug: "'ExpressionTree4CRF' object has no attribute 'num_nodes'"
		self.num_nodes = -1
		if self.num_nodes >= 0 and not recompute:
			return self.num_nodes

		self.num_nodes = 0

		node_stack = [[self.root,0]]
		while len(node_stack)>0:
			[current_node, current_child_id] = node_stack[-1]
			if current_child_id >= current_node.get_num_children():
				self.num_nodes += 1

				# remove node from stack
				node_stack.pop(-1)
			else:
				# increase child index:
				node_stack[-1][1] += 1
				# push child onto stack:
				node_stack.append([current_node.children[current_child_id], 0])

		return self.num_nodes

	@LM.logfunc
	def set_scope(self):
		nodes = self.root.get_leaves()
		while nodes:
			node = nodes.popleft()
			node.set_scope()
			if node.parent:
				nodes.append(node.parent)

	@LM.set_log_1
	@LM.logfunc_no_sublog
	def set_values(self, K, values, n):
		'''
		replace the variables with the specified indices with fixed parameters of the given values
		'''

		# construct maps that maps original var ids to marginal ids:
		id_map_disc = [-1 for _ in range(n)]
		current_map = 0
		for i in range(n):
			if i not in K:
				id_map_disc[i] = current_map
				current_map += 1

		# map the values to the positions of their indices:
		map_values = {}
		for k in range(len(K)):
			map_values[K[k]] = values[k]

		stack = deque([self.root])
		while len(stack) > 0:
			current_node = stack.pop()
			if len(current_node.children) > 0:
				for c in current_node.children:
					stack.append(c)
			elif isinstance(current_node, Node.Node_leaf_input):
				if current_node.var_id in map_values:
					if current_node.cat_value == map_values[current_node.var_id]:
						# add parameter node 1:
						param_node = Node.Node_leaf_param(parent=current_node.parent, theta=1)
						# remove node from parent:
						current_node.parent.children.remove(current_node)
					else:
						# add parameter node 0:
						param_node = Node.Node_leaf_param(parent=current_node.parent, theta=0)
						# remove node from parent:
						current_node.parent.children.remove(current_node)
				else:
					# adjust var id:
					current_node.var_id = id_map_disc[current_node.var_id]
					LM.log("adjust node id "+str(current_node)+": -> new node: "+str(current_node))	

	@LM.set_log_1
	@LM.logfunc_no_sublog
	def eval(self, input_disc):
		'''
		Evaluate the arithmetic circuit for a given input.

		input_disc: list of integers that describes the state of the (non-marginalized) categorical variables

		traverse graph post-order. each input leaf gets the coresponding value from the input-dict

		 node_stack contains pairs of [node, index]
		 where the index describes the next child of this node that has to be processed.
		 if index is larger then the number of children, then the node itself is processed.
		'''

		node_stack = self._construct_post_order(consider_leaves=True)
		for node in node_stack:
			if isinstance(node, Node.Node_leaf_input):
				#LM.log("is input leaf")
				node.set_value(input_disc[node.var_id])
						
			elif isinstance(node, Node.Node_internal):
				success = node.eval()
				if not success:
					# print graph if node failes to evaluate:
					print ("Error! Node "+str(node.id)+" failed to evaluate!")
			else:
				# case parameter node: nothing to do:
				pass

		return self.root.get_value()

	@LM.set_log_1
	@LM.logfunc
	def max(self):
		# get post-order of all internal nodes:
		node_stack = self._construct_post_order()

		# first pass: calculate the max evidence beginning from the leaves
		for current_node in node_stack:
			if isinstance(current_node, Node.Node_internal):
				current_node.maximize()
			else:
				# in case of input nodes or parameter nodes: do nothing
				pass
		
		# Second pass: Collecting the states
		max_states = []
		state_to_copies = {}
		work_list = [(self.root, [State()])]
		while work_list:
			current_node, states = work_list.pop(-1)
			# update the states if they were copied before
			_states = []
			for state in states:
				if state.id in state_to_copies.keys():
					updated_states = [state]
					while updated_states:
						up_state = updated_states.pop(-1)
						if up_state.id in state_to_copies.keys():
							for copied_state in state_to_copies[up_state.id]:
								updated_states.append(copied_state)
						else:
							_states.append(up_state)
					# states.remove(state)
				else:
					_states.append(state)
			states = _states

			if not current_node:
				continue
			# do something different for each node type
			if isinstance(current_node, Node.Node_internal):
				if current_node.expr_type == expr.ExpressionNodeType.ETYPE_MUL:
					# we follow each child
					for index, child in enumerate(current_node.children):
						work_list.append((child, states))
				elif current_node.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
					# we follow all children with max weight
					for index, child in enumerate(current_node.children):
						if np.isclose(child.max_weight, current_node.max_weight):
							new_states = []
							for state in states:
								new_state = state.clone()
								if state in max_states:
									max_states.remove(state)
								if state.id in state_to_copies:
									state_to_copies[state.id].append(new_state)
								else:
									state_to_copies[state.id] = [new_state]
								new_states.append(new_state)
							work_list.append((child, new_states))
			elif isinstance(current_node, Node.Node_leaf_input):
				for state in states:
					# collect all states we have to adjust
					if id(state) in state_to_copies.keys():
						updated_states = [state]
						while updated_states:
							up_state = updated_states.pop(-1)
							if up_state.id in state_to_copies.keys():
								for copied_state in state_to_copies[up_state.id]:
									if copied_state in max_states:
										max_states.remove(copied_state)
									copied_state.add_entry(current_node.var_id, current_node.cat_value)
									max_states.append(copied_state)
									updated_states.append(copied_state)
					else:
						if state in max_states:
							max_states.remove(state)
						state.add_entry(current_node.var_id, current_node.cat_value)
						max_states.append(state)

			elif isinstance(current_node, Node.Node_leaf_param):
				continue
			else:
				raise Exception("Node type known or not implemented.")

		# Remove dublicates
		max_states_tmp = []
		for i in max_states:
			if i.state == {}:
				continue
			if not i in max_states_tmp:
				max_states_tmp.append(i)
		max_states = max_states_tmp

		# create desired format
		return [tuple([assignment for var, assignment in sorted(state.state.items())]) for state in max_states]

	@LM.set_log_1
	@LM.logfunc
	def activate_variables(self, num_disc_total, disc_nodes_to_activate):
		'''
		Activates a subset of variables, i.e. replaced them parameter nodes with value 1
		Adjusts the indices of the remaining variables.

		Simply a sligthly adapted version of eval()
		'''
		# construct a map that maps discrete var ids to marginal ids:

		#print ("activate_variables")
		#print ("num_disc_total:",num_disc_total)
		#print ("disc_nodes_to_activate:",disc_nodes_to_activate)

		id_map = [-1 for _ in range(num_disc_total)]
		current_map = 0
		for i in range(num_disc_total):
			if i not in disc_nodes_to_activate:
				id_map[i] = current_map
				current_map += 1

		#print ("id_map:", id_map)

		node_stack = [self.root]
		while len(node_stack) > 0:
			current_node = node_stack[-1]
			node_stack.pop(-1)
			if isinstance(current_node, Node.Node_leaf_input):
				if current_node.var_id in disc_nodes_to_activate:
					# replace node with parameter node:
					#LM.log("activate node "+str(current_node))
					param_node = Node.Node_leaf_param(parent=current_node.parent, theta=1)
					current_node.parent.children.remove(current_node)
				else:
					# adjust variable id:
					#print ("var id:",current_node.var_id)
					current_node.var_id = id_map[current_node.var_id]
					#LM.log("adjust node id "+str(current_node)+": -> new node: "+str(current_node))
			else:
				for child in current_node.children:
					node_stack.append(child)

	

	def plot(self, filename="ac_tmp", full_labels=False):
		'''
		Does not plot parameter nodes
		if full_labels == False, node labels are simply node ids
			otherwise, full node labels are used (not yet implemented)
		'''
		# construct networkx tree:
		nxtree = nx.DiGraph()
		stack = [self.root]
		while len(stack) > 0:
			current = stack.pop(0)
			for c in current.children:
				if not isinstance(c, Node_leaf_param):
					nxtree.add_edge(current.id, c.id)
					stack.append(c)

		utils.plot_nx_graph(nxtree, filename, tree=True)


	def _construct_post_order(self, start_node=None, consider_leaves=False):
		'''
		Traverses the tree and constructs a post-order of all nodes in the Tree

		Return:
			a deque that contains all nodes of self in a valid post-order
		'''

		# set default start_node
		if start_node == None:
			start_node = self.root

		post_order = deque()
		down_stack = deque()
		down_stack.append(start_node)
		while len(down_stack) > 0:
			current_node = down_stack.pop()
			current_node.construct_substate()
			post_order.appendleft(current_node)
			for c in current_node.children:
				# add internal node to stack of nodes-to-process:
				if consider_leaves or isinstance(c, Node.Node_internal):
					down_stack.append(c)
		return post_order