#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

Classes for the different types of nodes required in arithmetic circuits
'''

from enum import Enum

import numpy as np
import math
from collections import deque

from compiler.ArithmeticCircuit import Expressions as expr

from compiler.util.LoggingManager import LoggingManager as LM

class Node:
	'''
	A base class for nodes of an AC

	Args:
		parent (Node):
			another node, this defines the internal structure of the tree

		label (String):
			a custom label given to this node, can be used for visualization or debugging

	Attributes:
		expr_type: only used for internal nodes. see below.

		children (list of ints):
			a list of the ids of all nodes that have this node as their parent. used for traversing the tree.

		value (float): only used for leaf nodes (leaf_input and leaf_parameter). see below

		id (int): a unique id to identify this node
	'''
	node_id = 0

	def __init__(self, parent=None, label=None):
		if parent:
			self.parent = parent
			parent.add_child(self)
		else:
			self.parent = None
		self.expr_type = expr.ExpressionNodeType.ETYPE_NONE
		self.label = label
		self.children = []
		self.value = None
		self.id = Node.node_id
		self.scope = set()
		self.substates = {}
		Node.node_id += 1
		# required for max algorithm: 
		self.max_states = {}
		self.max_weight = None

	def __str__(self):
		label = ""
		if not self.label:
			label += "("+str(self.id)+") "+expr.ETYPE_SHORT[self.expr_type]
		else:
			label += "("+str(self.id)+" - "+self.label+") "+expr.ETYPE_SHORT[self.expr_type]
		if not self.value is None:
			label += " = "+str(np.round(self.value,4))
		if not self.max_weight == None:
			label += " w="+str(self.max_weight)
		if self.max_states:
			label += f" ({str(self.max_states)})"
		return label

	def __repr__(self):
		return str(self)

	def to_dict(self):
		data = {
			'expr_type' : self.expr_type,
			'children' : [c.to_dict() for c in self.children]
		}
		return data

	def copy(self, new_parent=None):
		'''
		returns a deep copy of this node, but with a new parent.
		DOES NOT COPY CHILDREN!
		If parent == None, the original parent is used.
		'''
		if new_parent == None:
			new_parent = self.parent
		nodecopy = Node(parent=new_parent, label=self.label)
		nodecopy.scope = set(self.scope)
		nodecopy.max_states = {s: self.max_states[s] for s in self.max_states}
		nodecopy.max_weight = self.max_weight
		return nodecopy

	@LM.logfunc
	def copy_with_children(self, new_parent=None):
		'''
		returns a deep copy of this node AND ALL DESCENDING NODES, but with a new parent.
		If parent == None, the original parent is used.
		'''

		nodecopy = self.copy(new_parent)
		# descendants_stack contains pairs of [copy, original]
		descendants_stack = [[nodecopy, self]]
		while len(descendants_stack) > 0:
			[desc_copy, desc_orig] = descendants_stack[-1]
			descendants_stack.pop(-1)
			for c in desc_orig.children:
				desc_child_copy = c.copy(desc_copy)
				descendants_stack.append([desc_child_copy, c])

		return nodecopy

	def get_description(self):
		'''
		constructs a longer string that describes this node in detail.
		'''
		str_ = "Node "+str(self.id)
		if self.label:
			str_ += " : "+str(self.label)
		str_ += " ("+str(self.expr_type)+")"
		if self.parent:
			str_ += " is child of "+str(self.parent.id)
		if len(self.children) > 0:
			str_ += "\n\thas children: "+str([c.id for c in self.children])
		if self.scope:
			str_ += "\n\thas scope: "+str(self.scope)
		else:
			str_+= "\n\thas no scope set"
		if self.substates:
			str_ += "\n\thas substates: "
			for x in self.substates:
				str_ += "\n\t\t"+str(x)+" - "+str(self.substates[x])
		else:
			str_ += "\n\thas no substates"
		return str_

	def scope(self):
		return self.scope

	def add_child(self, child):
		self.children.append(child)

	def is_root(self):
		return not self.parent

	def get_value(self):
		return self.value

	def get_num_children(self):
		return len(self.children)

	def set_parent(self, new_parent, force=False):
		'''
		new_parent (Node): the new parent node
		force (bool) : if False, this WILL FAIL if
			this node is not in the list of children of its current parent
			(in most cases, this means that something somewhere else broke the tree structure)
		'''
		# remove this as child from current parent:
		if self.parent:
			if not force:
				self.parent.children.remove(self)
			elif self in self.parent.children:
				self.parent.children.remove(self)
		# add to new parent:
		new_parent.add_child(self)
		self.parent = new_parent

	def get_leaves(self):
		'''
		traverses through the tree, returns all nodes that do not have children
		'''
		leaves = deque([])
		for child in self.children:
			for i in child.get_leaves():
				leaves.append(i)
		if not self.children:
			return [self]
		return leaves

	def construct_substate(self):
		if not self.parent == None:
			for i in self.parent.substates:
				self.add_to_substate(i, self.parent.substates[i])
		for child in self.children:
			if isinstance(child, Node_leaf_input):
				self.add_to_substate(child.var_id, child.cat_value)
				# explicitly construct substate of child
				# because input leaves are not considered during simplify:
				child.construct_substate()

	def max(self):
		# do nothin in default case.
		# is implemented for Node_internal
		return

	def add_to_substate(self, var_id, value):
		self.substates[var_id] = value

	def set_scope(self):
		raise NotImplementedError("Has to be done in the from 'Node' derived classes.")

	def simplify(self):
		raise NotImplementedError("Has to be done in the from 'Node' derived classes.")

	def eval(self):
		raise NotImplementedError("Has to be done in the from 'Node' derived classes.")

	def _map_eval(self, conditions, missing_map_vars):
		raise NotImplementedError("Has to be done in the from 'Node' derived classes.")

class Node_internal(Node):
	'''
	A internal node of an ET that computes an arbitrary function that takes either an input of a specific dimension

	Args:
		expr_type: has to be one of these (specified in InferenceEngine.DataStructure.ExpressionTree.Expressions):
		- ETYPE_MUL
		- ETYPE_ADD
		- ETYPE_NONE (not to be used, only for testing)
		this is used to apply a valid function that this node implements.

	Attributes:
		expression: one of a pre-defined function that takes a list of floats and ints and returns a float
			(as specified in the dictionary "EXPRESSIONS", see InferenceEngine.DataStructure.ExpressionTree.Expressions).
	'''
	def __init__(self, expr_type, parent=None, label=None):
		super().__init__(parent, label)
		self.expr_type = expr_type
		self.expression = expr.EXPRESSIONS[expr_type]

	def copy(self, new_parent):
		'''
		returns a deep copy of this node, but with a new parent.
		If parent == None, the original parent is used.
		'''
		if new_parent == None:
			new_parent = self.parent
		nodecopy = Node_internal(expr_type=self.expr_type, parent=new_parent, label=self.label)
		nodecopy.scope = set(self.scope)
		nodecopy.max_states = {s: self.max_states[s] for s in self.max_states}
		nodecopy.max_weight = self.max_weight
		return nodecopy

	def to_dict(self):
		data = super().to_dict()
		data['class'] = "INTERNAL"
		return data

	def is_valid(self):
		return len(self.children) > 0

	def set_scope(self):
		scope = set()
		for child in self.children:
			scope = scope.union(child.scope)
		self.scope = scope

	@LM.logfunc
	def check_if_merge_node(self, i, j):
		'''
		check if this node is a merge-node of x_i and x_j
		'''
		if not self.expr_type == expr.ExpressionNodeType.ETYPE_MUL:
			# merge node has to be a product-node
			return False
		elif not i in self.scope or not j in self.scope:
			# merge node has to contain both variables in its scope:
			return False
		else:
			for c in self.children:
				if i in c.scope or j in c.scope:
					if isinstance(c, Node_leaf_input):
						# at least one variable is a direct child: not a merge node
						return False
					if i in c.scope and j in c.scope:
						# both variables are in the scope of a single child: not a merge node
						return False
			return True

	#@LM.logfunc
	@LM.set_log_0
	def simplify(self):
		LM.log("simplify node "+str(self.id)+" ("+expr.ETYPE_SHORT[self.expr_type]+")")

		# expression-type specific simplifications:
		if self.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
			node_still_exists = self._simplify_sum()
		elif self.expr_type == expr.ExpressionNodeType.ETYPE_MUL:
			node_still_exists = self._simplify_prod()

		if node_still_exists:
			# contract node if 0 or 1 children left:
			if len(self.children) == 0:
				# if node has no children, remove it:
				LM.log("remove node without children")
				if self.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
					self._replace_with_param(0)
				else:
					self._replace_with_param(1)
			elif len(self.children) == 1:
				# if node has only one children, contract node, ie replace this by its child
				LM.log("contract node: replace with its single child")
				if not self.parent == None:
					# cannot replace root because root is fixed
					self.children[0].set_parent(self.parent)
					self.parent.children.remove(self)

	@LM.logfunc
	def _simplify_sum(self):
		'''
		Otherwise:
		 contracts all parameter children of node into one parameter
		if all childlen of this node are parameters, contract this with all children into single parameter
		if node has no children, remove it

		returns True, if self is still part of the ET after simplification, False otherwise
		'''
		# at first: check if there are non-alternating children:
		children = [c for c in self.children]
		for c in children:
			if isinstance(c, Node_internal) and c.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
				# node is inflated: add all children of c to this node, then remove c
				LM.log("child "+str(c)+" is inflated! at its children to self and remove "+str(c))
				grandchildren = [cc for cc in c.children]
				for gc in grandchildren:
					gc.set_parent(self)
					LM.log(" Add new child "+str(gc))
				self.children.remove(c)

		# combine all parameter-children into one:
		all_children_parameters = True
		sum_ = 0
		children = [c for c in self.children]
		for c in children:
			if isinstance(c, Node_leaf_param):
				# remove this child:
				self.children.remove(c)
				# increase total sum of child-parameters:
				sum_ += c.value
			else:
				all_children_parameters = False

		if all_children_parameters:
			# replace this node with a fixed parameter:
			LM.log("replace with parameter")
			self._replace_with_param(sum_)
			return False
		elif sum_ != 0:
			LM.log("merge parameters")
			param_node = Node_leaf_param(parent=self, theta=sum_)
		return True

	@LM.logfunc
	def _simplify_prod(self):
		'''
		contracts all parameter children of node into one parameter
		if all chidlren of this node are parameters, contract this with all children into single parameter
		if this node has multiple exp-nodes as children, these are merged into an exp of a sum

		returns True, if self is still part of the ET after simplification, False otherwise
		'''

		# at first: check if there are non-alternating children:
		children = [c for c in self.children]
		for c in children:
			if isinstance(c, Node_internal) and c.expr_type == expr.ExpressionNodeType.ETYPE_MUL:
				# node is inflated: add all children of c to this node, then remove c
				LM.log("child "+str(c)+" is inflated! at its children to self and remove "+str(c))
				grandchildren = [cc for cc in c.children]
				for gc in grandchildren:
					gc.set_parent(self)
					LM.log(" Add new child "+str(gc))
				self.children.remove(c)

		# combine all parameter-children into one, and also all exponentiated children into an exponentiated sum:
		still_exists = True
		all_children_parameters = True
		prod_ = 1
		children = [c for c in self.children]
		for c in children:
			if isinstance(c, Node_leaf_param):
				# remove this child:
				self.children.remove(c)
				# increase total sum of child-parameters:
				prod_ *= c.value
			else:
				all_children_parameters = False

		if all_children_parameters:
			# replace this node with a fixed parameter:
			LM.log("replace with param")
			self._replace_with_param(prod_)
			still_exists = False
		elif prod_ == 0:
			LM.log("replace with 0")
			self._replace_with_param(prod_)
			still_exists = False
		elif prod_ != 1:
			LM.log("merge parameters")
			param_node = Node_leaf_param(parent=self, theta=prod_)
		return still_exists

	def maximize(self):
		# expression-type specific simplifications:
		if self.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
			self._maximize_sum()
		elif self.expr_type == expr.ExpressionNodeType.ETYPE_MUL:
			self._maximize_prod()

	def _maximize_sum(self):
		weights = [child.max_weight for child in self.children]
		# SINCE the GRAPH is DETERMINISTIC ONE of the CHILDREN has the HIGHEST WEIGT
		self.max_weight = max(weights)

	def _maximize_prod(self):
		prod = 1.0
		for child in self.children:
			prod *= child.max_weight
		self.max_weight = prod

	@LM.logfunc
	def _replace_with_param(self, param):
		if not self.parent == None:
			# cannot replace root because root is fixed
			param_node = Node_leaf_param(parent=self.parent, theta=param)
			LM.log("Replace with new node: "+str(param_node))
			self.parent.children.remove(self)
		else:
			param_node = Node_leaf_param(parent=self, theta=param)
			LM.log("Since node is root, no replacing happens.")

	@LM.logfunc
	def _contract_with_child(self, param):
		# remove node from parent:
		if not self.parent == None:
			self.parent.children.remove(self)
		# add new (merged) parameter node:
		merged_node = Node_leaf_param(parent=self.parent, theta=param)
		if self.label:
			merged_node.label = self.label

	@LM.logfunc
	def eval(self):
		input_ = [c.get_value() for c in self.children]
		LM.log("node id: "+str(self.id)+", input: "+str(input_))
		self.value = self.expression(input_)
		if self.value == None:
			print ("Error: None value in node:")
			print (self.get_description())
			return False
		else:
			return True

	@LM.logfunc
	def _map_eval(self, conditions, missing_map_vars):
		if self.expr_type == ExpressionNodeType.ETYPE_MUL:
			prod = 1.0
			for child in self.children:
				prod *= child._map_eval(conditions, missing_map_vars)
			return prod
		elif self.expr_type == ExpressionNodeType.ETYPE_ADD:
			if any([i in self.scope for i in missing_map_vars]):
				return np.max([child._map_eval(conditions, missing_map_vars) for child in self.children])
			else:
				return np.sum([child._map_eval(conditions, missing_map_vars) for child in self.children])
		else:
			raise Exception("Node type not allowed.")

	@LM.logfunc
	def max(self):
		# assumes that all children have max_weight and max_states already computed.
		# DOES NOT WORK CORRECTLY IN CURRENT STATE
		raise NotImplementedError()

		self.max_states = {}
		if self.expr_type == expr.ExpressionNodeType.ETYPE_MUL:
			self.max_weight = 1
			for c in self.children:
				self.max_weight *= c.max_weight
				for s in c.max_states:
					self.max_states[s] = c.max_states[s]
		elif self.expr_type == expr.ExpressionNodeType.ETYPE_ADD:
			current_max_weight = None
			current_max_states = {}
			for c in self.children:
				if current_max_weight == None or current_max_weight < c.max_weight:
					current_max_weight = c.max_weight
					current_max_states = c.max_states
				elif np.isclose(current_max_weight, c.max_weight):
					for var in c.max_states:
						for assignment in c.max_states[var]:
							if not assignment in current_max_states[var]:
								current_max_states[var].append(assignment)
					current_max_weight = c.max_weight
			self.max_weight = current_max_weight
			self.max_states = current_max_states
		else:
			raise Exception("Node type not allowed.")
		
class Node_leaf_input(Node):
	'''
	A leaf node that represents an input variable

	Args:
		var_id (int):
			the index of the continuous (or categorical) variable (cont or cat depends on vtype).
			Is used to identify the corresponging variable of the CG-distribution.
		vtype (one of 'TYPE_CONTV' and 'TYPE_CAT', as specified in NODE_TYPES (see aove)):
			specifies whether this node represents a continuous variable or a discrete variable
		cat_value (int):
			if this node represents a categorical variable, it is actually a indicator variable for 
			one specific value of the cat. variable.
			cat_value defines this value that this node should indicate.
	'''
	def __init__(self, parent, var_id, cat_value=-1, label=None):
		super().__init__(parent, label)
		self.var_id = var_id
		if cat_value<0:
			# TODO: raise exception: cat_value not set
			self.cat_value = 0
		self.cat_value = cat_value
		self.scope.add(self.var_id)
		self.add_to_substate(var_id, cat_value)
		parent.add_to_substate(var_id, cat_value)

		self.max_states = {var_id : [cat_value]}
		self.max_weight = 1

	def __str__(self):
		return "("+str(self.id)+") X_"+str(self.var_id)+"="+str(self.cat_value)
		
	def copy(self, new_parent):
		'''
		returns a deep copy of this node, but with a new parent.
		If parent == None, the original parent is used.
		'''
		if new_parent == None:
			new_parent = self.parent
		nodecopy =  Node_leaf_input(parent=new_parent, var_id=self.var_id, cat_value=self.cat_value, label=self.label)
		nodecopy.scope = set(self.scope)
		nodecopy.max_states = {s: self.max_states[s] for s in self.max_states}
		nodecopy.max_weight = self.max_weight
		return nodecopy

	def to_dict(self):
		data = super().to_dict()
		data['class'] = "LEAF_INPUT"
		data['var_id'] = self.var_id
		data['cat_value'] = self.cat_value
		return data

	def get_description(self):
		str_ = "Leaf "+str(self.id)+" (id: "+str(self.var_id)+")"
		if self.parent:
			str_ += " is child of "+str(self.parent.id)
		return str_

	def set_value(self, x):
		if x == self.cat_value:
			# set to 1 if input has correct value, otherwise 0
			self.value = 1
		else:
			self.value = 0

	def activate(self):
		# makes a node always return 1
		# required for computing distributions with marginalized discrete vars
		self.value = 1	
		
	def set_scope(self):
		self.scope = set()
		self.scope.add(self.var_id)

	@LM.logfunc
	def _map_eval(self, conditions, missing_map_vars):
		if self.var_id in conditions and conditions[self.var_id] == self.cat_value:
			return 1
		else:
			if not self.var_id in conditions:
				return 1
			return 0

	def _simplify(self):
		'''
		Do nothing.
		'''
		pass

class Node_leaf_param(Node):
	'''
	A leaf node that contains a fixed parameter theta of the distribution

	Args:
		theta (float):
			a parameter of the CG distribution that is set as the fixed value of this node.
	'''
	def __init__(self, theta, parent, label=None):
		super().__init__(parent, label)
		self.value = theta

		self.max_states = {}
		self.max_weight = self.value

	def __str__(self):
		if not isinstance(self.value, np.ndarray):
			return "("+str(self.id)+") "+str(round(self.value,4))
		else:
			return "("+str(self.id)+") ndarray"+str(self.value.shape)

	def __repr__(self):
		return self.__str__()

	def copy(self, new_parent):
		'''
		returns a deep copy of this node, but with a new parent.
		If parent == None, the original parent is used.
		'''
		if new_parent == None:
			new_parent = self.parent

		nodecopy = Node_leaf_param(theta=self.value, parent=new_parent, label=self.label)
		nodecopy.scope = set(self.scope)
		nodecopy.max_states = {s: self.max_states[s] for s in self.max_states}
		nodecopy.max_weight = self.max_weight
		return nodecopy

	def to_dict(self):
		data = super().to_dict()
		data['class'] = "LEAF_PARAM"
		data['value'] = self.value
		return data

	def get_description(self):
		str_ = "Leaf "+str(self.id)+" (PARAM: "+str(self.value)+")"
		if self.parent:
			str_ += " is child of "+str(self.parent.id)
		return str_

	@LM.logfunc
	def eval(self):
		return self.value

	def set_scope(self):
		self.scope = set()

	@LM.logfunc
	def _map_eval(self, conditions, missing_map_vars):
		return self.value

	def _simplify(self):
		'''
		Do nothing.
		'''
		pass