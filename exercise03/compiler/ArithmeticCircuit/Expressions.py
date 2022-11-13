#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

Basics needed in arithmetic circuits
'''

from enum import Enum

import numpy as np
import math

def expr_mul(input):
	'''
	takes a list of arbitrary length as input
	'''
	result = 1
	for x in input:
		result *= x
	return result

def expr_add(input):
	'''
	takes a list of arbitrary length as input
	'''
	result = 0
	for x in input:
		result += x
	return result

# EXPRESSION NODE TYPES:
class ExpressionNodeType(Enum):
	ETYPE_MUL  = 0 #"MULTIPLICATION"
	ETYPE_ADD  = 1 #"ADDITION"
	ETYPE_NONE = 3 #"NONE"

EXPRESSIONS = {
	ExpressionNodeType.ETYPE_MUL : expr_mul,
	ExpressionNodeType.ETYPE_ADD : expr_add
}

ETYPE_SHORT = {
	ExpressionNodeType.ETYPE_MUL : '*',
	ExpressionNodeType.ETYPE_ADD : '+'
}
