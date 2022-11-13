#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
@author: Andreas Goral

Logging features
'''

import logging
import datetime
from pathlib import Path

from compiler.util.org import DIR_LOG
from compiler.util.utils import ensure_path_exists

class LoggingManager:
	'''
	main functionalities of Logging Manager :
	1) logging:
		- logfunc: a function wrapper that writes to logging.info whenever the wrapped function is entered and finished
		- log: a function wrapper for logging.info
		- logvar: a function wrapper to log variable states to logging.info
	2) different log-levels
		2: all logging
		1: only logfunc
		0: none
	3) a global dotted indent that is 
		- increased by one dot whenever a logged function (i.e. a function wrapped by logfunc) is entered,
		- and decreased again when the function is finished
	'''
	__current_indent = ""
	__log_level = 2 #2: all; 1: only logfunc; 0: None
	__sub_log_level = 2

	def init_log(message=None, datetime_in_filename=True):
		ensure_path_exists(DIR_LOG)
		if datetime_in_filename:
			if not message == None:
				logfile = f"LOG_{datetime.datetime.now().isoformat(timespec='seconds').replace(':', '-')}_{message}.log"
			else:
				logfile = f"LOG_{datetime.datetime.now().isoformat(timespec='seconds').replace(':', '-')}.log"
		else:
			if not message == None:
				logfile = f"LOG_{message}.log"
			else:
				logfile = f"LOG_tmp.log"
		print ("Init logging into file: "+logfile)
		filepath = DIR_LOG / logfile
		if filepath.exists():
			filepath.unlink()
		logging.basicConfig(filename=str(filepath), level=logging.INFO)
		logging.info(f"Start logging at {datetime.datetime.now().isoformat(timespec='seconds')}.")
		if not message == None:
			logging.info(message)

	def logfunc(func):
		'''
		decorator that logs start of function and increases indentation
		'''
		def inner(*args, **kwargs):
			# log function start:
			if LoggingManager.__log_level > 0:
				logging.info(datetime.datetime.now().isoformat(timespec='seconds')+LoggingManager.__current_indent+" Entering function: "+func.__qualname__)
				# increase indent:
				LoggingManager.__current_indent += "."
			# decrease log level if sub_log_level is smaller than current log level
			previous_log_level = LoggingManager.__log_level
			if LoggingManager.__sub_log_level < LoggingManager.__log_level:
				LoggingManager.__log_level = LoggingManager.__sub_log_level
			try:
				return_ = func(*args, **kwargs)
				# reset log level:
				LoggingManager.__log_level = previous_log_level
				# reduce indent:
				if LoggingManager.__log_level > 0:
					LoggingManager.__current_indent = LoggingManager.__current_indent[:-1]
					#logging.info(LoggingManager.__current_indent+" Finished function: "+func.__qualname__)
				return return_
			except:
				# reset log level:
				LoggingManager.__log_level = previous_log_level
				# reduce indent:
				if LoggingManager.__log_level > 0:
					LoggingManager.__current_indent = LoggingManager.__current_indent[:-1]
				raise

		return inner

	def logfunc_no_sublog(func):
		'''
		decorator that logs start of function and increases indentation
		disables logging for all functions called from within func
		'''
		def inner(*args, **kwargs):
			# log function start:
			if LoggingManager.__log_level > 0:
				logging.info(datetime.datetime.now().isoformat(timespec='seconds')+LoggingManager.__current_indent+" Entering function: "+func.__qualname__)
				# increase indent:
				LoggingManager.__current_indent += "."
			# decrease log level if sub_log_level is smaller than current log level
			previous_log_level = LoggingManager.__log_level
			if LoggingManager.__sub_log_level < LoggingManager.__log_level:
				LoggingManager.__log_level = LoggingManager.__sub_log_level

			# set sub_log_level for further functions:
			previous_sub_log_level = LoggingManager.__sub_log_level
			LoggingManager.__sub_log_level = 0
			try:
				return_ = func(*args, **kwargs)
				# reset log level:
				LoggingManager.__log_level = previous_log_level

				# reset sub_log_level:
				LoggingManager.__sub_log_level = previous_sub_log_level
				# reduce indent:
				if LoggingManager.__log_level > 0:
					LoggingManager.__current_indent = LoggingManager.__current_indent[:-1]
					#logging.info(LoggingManager.__current_indent+" Finished function: "+func.__qualname__)
				return return_
			except:
				# reset log level:
				LoggingManager.__log_level = previous_log_level
				# reset sub_log_level:
				LoggingManager.__sub_log_level = previous_sub_log_level

				# reduce indent:
				if LoggingManager.__log_level > 0:
					LoggingManager.__current_indent = LoggingManager.__current_indent[:-1]
				raise

		return inner

	def set_log_0(func):
		'''
		decorator that sets logging-level to 0 within the function:
		'''
		def inner(*args, **kwargs):
			previous_log_level = LoggingManager.__log_level
			LoggingManager.__log_level = 0
			try:
				return_ = func(*args, **kwargs)
				LoggingManager.__log_level = previous_log_level
				return return_
			except:
				LoggingManager.__log_level = previous_log_level
				raise
		return inner

	def set_log_1(func):
		'''
		decorator that sets logging-level to 1 within the function:
		'''
		def inner(*args, **kwargs):
			previous_log_level = LoggingManager.__log_level
			if previous_log_level > 1:
				LoggingManager.__log_level = 1
			try:
				return_ = func(*args, **kwargs)
				LoggingManager.__log_level = previous_log_level
				return return_
			except:
				LoggingManager.__log_level = previous_log_level
				raise
		return inner

	def set_log_2(func):
		'''
		decorator that sets logging-level to 2 within the function:
		'''
		def inner(*args, **kwargs):
			previous_log_level = LoggingManager.__log_level
			LoggingManager.__log_level = 2
			try:
				return_ = func(*args, **kwargs)
				LoggingManager.__log_level = previous_log_level
				return return_
			except:
				LoggingManager.__log_level = previous_log_level
				raise
		return inner

	def log(message):
		if LoggingManager.__log_level > 1:
			logging.info(datetime.datetime.now().isoformat(timespec='seconds')+LoggingManager.__current_indent+" "+message)

	def logvar(name, value):
		if LoggingManager.__log_level > 1:
			logging.info(datetime.datetime.now().isoformat(timespec='seconds')+LoggingManager.__current_indent+" LOGVAR: "+name + " = " + str(value))