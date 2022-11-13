#!usr/bin/python
# -*- coding: utf-8 -*-import string

'''
Organization of file structure
'''
from pathlib import Path

# main directory for temporary and debug output:
DIR_OUT = Path("tmp")

# main directoy for expression trees and other graphviz output:
DIR_GRAPH = DIR_OUT/ "graphviz"

# main directory for nx-graphplots:
DIR_NX = DIR_OUT / "nx"

# main directory for cnf and nnf files (tmp files required during compilation):
DIR_NNF = DIR_OUT / "nnf"


# main directory for experiment-related data:
DIR_EXP = Path("data")

# main directory for models:
DIR_MOD = DIR_EXP / "models"

# main directory for experiment results and configurations:
DIR_EXP_RES = DIR_EXP / "experiments"


# main directory for logs:
DIR_LOG = Path("logs")


# main directory for datasets:
DIR_DATA = DIR_EXP / "datasets"