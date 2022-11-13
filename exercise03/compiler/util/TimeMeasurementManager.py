#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

Time measurement decorator
'''

from time import time

class TimeMeasurementManager:
	'''
	Decorators to measure runtime of marg, cond, eval and max
	'''
	__is_active = False
	__is_measuring_eval = False
	__is_measuring_max = False
	__time_condition = -1
	__time_marg_cont = -1
	__time_marg_disc = -1
	__time_eval = -1
	__time_max  = -1

	def init():
		TimeMeasurementManager.__is_active = True
		TimeMeasurementManager.__time_condition = -1
		TimeMeasurementManager.__time_marg_cont = -1
		TimeMeasurementManager.__time_marg_disc = -1
		TimeMeasurementManager.__time_eval = -1
		TimeMeasurementManager.__time_max  = -1

	def set_zeros():
		'''
		sets time_condition, time_marg_cont, and time_marg disc to 0
		needed for brute-force-measurements where these times should be ignored.
		'''
		TimeMeasurementManager.__time_condition = 0
		TimeMeasurementManager.__time_marg_cont = 0
		TimeMeasurementManager.__time_marg_disc = 0

	def time_condition(func):
		'''
		decorator that logs time of conditioning
		'''
		def inner(*args, **kwargs):
			if TimeMeasurementManager.__is_active:
				time_start = time()
			return_ = func(*args, **kwargs)
			if TimeMeasurementManager.__is_active:
				if TimeMeasurementManager.__time_condition < 0:
					TimeMeasurementManager.__time_condition = 0
				TimeMeasurementManager.__time_condition += time()-time_start
			return return_
		return inner

	def time_marg_cont(func):
		'''
		decorator that logs time of marginalizing continuous variables
		'''
		def inner(*args, **kwargs):
			if TimeMeasurementManager.__is_active:
				time_start = time()
			return_ = func(*args, **kwargs)
			if TimeMeasurementManager.__is_active:
				if TimeMeasurementManager.__time_marg_cont < 0:
					TimeMeasurementManager.__time_marg_cont = 0
				TimeMeasurementManager.__time_marg_cont += time()-time_start
			return return_
		return inner

	def time_marg_disc(func):
		'''
		decorator that logs time of marginalizing discrete variables
		'''
		def inner(*args, **kwargs):
			if TimeMeasurementManager.__is_active:
				time_start = time()
			return_ = func(*args, **kwargs)
			if TimeMeasurementManager.__is_active:
				if TimeMeasurementManager.__time_marg_disc < 0:
					TimeMeasurementManager.__time_marg_disc = 0
				TimeMeasurementManager.__time_marg_disc += time()-time_start
			return return_
		return inner

	def time_eval(func):
		'''
		decorator that logs time of evaluating the density
		'''
		def inner(*args, **kwargs):
			# check if time_eval is currently running
			# (ensures that recursive calls are not measured multiple times)
			measure_this = not TimeMeasurementManager.__is_measuring_eval
			if TimeMeasurementManager.__is_active and measure_this:
				time_start = time()
				TimeMeasurementManager.__is_measuring_eval = True
			return_ = func(*args, **kwargs)
			if TimeMeasurementManager.__is_active and measure_this:
				if TimeMeasurementManager.__time_eval < 0:
					TimeMeasurementManager.__time_eval = 0
				TimeMeasurementManager.__time_eval += time()-time_start
				TimeMeasurementManager.__is_measuring_eval = False
			return return_
		return inner

	def time_max(func):
		'''
		decorator that logs time of maximizing
		'''
		def inner(*args, **kwargs):
			# check if time_max is currently running
			# (ensures that recursive calls are not measured multiple times)
			measure_this = not TimeMeasurementManager.__is_measuring_max
			if TimeMeasurementManager.__is_active and measure_this:
				time_start = time()
				TimeMeasurementManager.__is_measuring_max = True
			return_ = func(*args, **kwargs)
			if TimeMeasurementManager.__is_active and measure_this:
				if TimeMeasurementManager.__time_max < 0:
					TimeMeasurementManager.__time_max = 0
				TimeMeasurementManager.__time_max += time()-time_start
				TimeMeasurementManager.__is_measuring_max = False
			return return_
		return inner

	def get_times():
		times = {
			'cond' : TimeMeasurementManager.__time_condition,
			'marg_disc' : TimeMeasurementManager.__time_marg_disc,
			'marg_cont' : TimeMeasurementManager.__time_marg_cont,
			'eval' : TimeMeasurementManager.__time_eval,
			'max' : TimeMeasurementManager.__time_max
		}
		return times