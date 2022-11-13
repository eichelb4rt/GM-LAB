
#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral
'''

class DimensionMismatchException(Exception):
	'''
	Exception that should be raised whenever the input arguments of a function have somehow mismatching dimensions
	'''
	def __init__(self, var_name, d1, d2):
		message = "Dimension for variable "+var_name+" do not fit: "+str(d1)+" != "+str(d2)
		super().__init__(message)