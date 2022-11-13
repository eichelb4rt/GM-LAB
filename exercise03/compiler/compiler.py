## Author: Julien Klaus
## Email: julien.klaus@uni-jena.de

import platform
import subprocess
from numbers import Number
import itertools as it
from time import time
from collections import deque

from pathlib import Path
import numpy as np

from compiler.ArithmeticCircuit.Expressions import ExpressionNodeType as ENT
from compiler.ArithmeticCircuit.Node import Node_internal, Node_leaf_input, Node_leaf_param
from compiler.ArithmeticCircuit.ArithmeticCircuit import ArithmeticCircuit

from compiler.util.LoggingManager import LoggingManager as LM
from compiler.util.org import DIR_NNF, DIR_GRAPH
from compiler.util.utils import ensure_path_exists

@LM.logfunc
def compile_discrete_interaction(q, interactions=2, verbose=False, var_ids=None, unique_name=None):
    '''
    Args:
        var_ids (list of integers)(optional):
        A list of variable ids that are used in the construction of the expression tree.
        If None, variable ids will be 0,1,...
    '''

    if interactions != 2:
        raise Exception("Until now only pairwise interactions are allowed.")

    if unique_name == None:
        filename = "graphical_model.cnf"
    else:
        filename = unique_name+".cnf"

    ####### REMOVE THIS LATER
    if verbose:
        print ("WARNING: VERBOSE ACTIVATED!")
    ####### THIS LINE IST JUST FOR A SHORT DEBUG

    if verbose:
        start = time()

    start = time()
    if verbose:
        print("Start discrete compilation")
    LM.log("Start discrete compilation")

    # create CNF
    lf = LogicalFormula()
    cnf_list = lf.get_cnf(q, verbose=verbose)
    if verbose:
        print("Create CNF in ..", end=" ")
        print(f"{time() - start}s")
    LM.log(f"Create CNF in .. {time() - start}s")

    instance_to_number = lf.inst_to_number
    parameter_values = lf.par_values
    cnf_object = CNF(cnf_list)
    ctet = CnfToExpTree()

    if verbose:
        print("Compile into Darwiche structure in ..", end=" ")
    LM.log("Compile into Darwiche structure in ..")
    start = time()

    # create logical tree out of cnf
    root = ctet.compile(cnf_object, simplify=True, num_to_var=instance_to_number, filename=filename)

    if verbose:
        print(f"{time() - start}s")
        print("Create expression tree in ..", end=" ")
    LM.log(f"{time() - start}s")
    LM.log(f"Create expression tree in ..")
    start = time()

    # translate into expression tree (internal)
    exp_tree = ExprTree()
    exp_tree.create(root, instance_to_number, parameter_values, var_ids)

    if verbose:
        print(f"{time() - start}s")
        print("Simplify expression tree in ..", end=" ")
    LM.log(f"{time() - start}s")
    LM.log(f"Simplify expression tree in .. ")
    start = time()

    # simplify this tree (removing one nodes)
    exp_tree.simplify()

    if verbose:
        print(f"{time() - start}s")
        print("Translate expression tree into ET in ..", end=" ")
    LM.log(f"{time() - start}s")
    LM.log(f"Translate expression tree into ET in .. ")
    start = time()

    # translate the internal expression tree into a ET from Andreas
    # reverse var_ids:
    id_to_var_ids = {}
    for i in range(len(var_ids)):
        id_to_var_ids[var_ids[i]] = i
    tree = exp_tree.create_ET(instance_to_number, parameter_values, q, id_to_var_ids)

    if verbose:
        print(f"{time() - start}s")
    LM.log(f"{time()-start}s")
    start = time()

    if verbose:
        print ("Compile finished!")
    return tree

@LM.logfunc
def nnf_file_to_graph(cnf_file="graphical_model.cnf"):
    ensure_path_exists(DIR_NNF)
    nnf_file = Path(f"{DIR_NNF}/{cnf_file}.nnf")
    with open(nnf_file, "r+") as file:
        lines = file.readlines()
    # remove ending "\n"
    lines = [line.rstrip("\n") for line in lines]
    # first line contains many informations
    tag, n_nodes, n_edges, n_vars = lines[0].split(" ")
    assert tag == "nnf", f"NNF file suspected, got {tag} file"
    # the tree is build from the end to the top
    indexed_nodes = dict(enumerate(lines[1:]))
    created_nodes = indexed_nodes.copy()
    for index, node in indexed_nodes.items():
        elements = node.split(" ")
        # OR Node (label, j (see manual), n_children, c1, c2, ...)
        if elements[0] == "O":
            # false node
            if elements[1] == 0:
                created_nodes[index] = NNFNode("false", "false", index)
            else:
                or_node = NNFNode("O", "or", index)
                for index_tmp in range(int(elements[2])):
                    child_index = int(elements[3 + index_tmp])
                    or_node.add_child(created_nodes[child_index])
                created_nodes[index] = or_node
        # AND Node (label, n_children, c1, c2, ..)
        elif elements[0] == "A":
            and_node = NNFNode("A", "and", index)
            for index_tmp in range(int(elements[1])):
                child_index = int(elements[2 + index_tmp])
                and_node.add_child(created_nodes[child_index])
            created_nodes[index] = and_node
        # Leaf Node (label, variable)
        elif elements[0] == "L":
            created_nodes[index] = NNFNode("L", int(elements[1]), index)
        else:
            raise NameError(f"Node type not expected: {elements}")

    return created_nodes[int(n_nodes) - 1]

class LogicalFormula(object):
    def __init__(self):
        pass

    def _var_state_combination_valid(self, vars, states):
        # if we have the same variable we just need the main diagonal
        if all(elem == vars[0] for elem in vars):
            # states equal?
            if all(st == states[0] for st in states):
                return True
        # check if we have different vars
        else:
            if len(np.unique(vars)) == len(vars):
                return True
            else:
                raise Exception("Different Interactions size then 2. Rewrite this function!")
        return False

    def _vars_equal(self, vars):
        if all(elem == vars[0] for elem in vars):
            return True
        return False

    def _create_exclusive_clauses(self, var_states, running_var_number=1):
        """
        Creates all exclusive clauses for variables and states
        """
        exclusive_clauses = []
        variables = {}
        for var, states in var_states.items():
            # first a clause with all positive
            disjunction_all_true = []
            for state in states:
                disjunction_all_true.append(running_var_number)
                variables[(var, state)] = running_var_number
                running_var_number += 1
            exclusive_clauses.append(disjunction_all_true)
            # second clauses with all pairwise negations
            for s1, s2 in it.combinations(states, 2):
                exclusive_clauses.append([-variables[(var, s1)], -variables[(var, s2)]])
        return exclusive_clauses, running_var_number, variables

    def _construct_cnf(self, q, verbose=False):
        parameter_vars = {}
        cnf = []
        ds, var_states = self._build_data_structure(q)

        ### exclusive variable states ###
        exclusive_clauses, running_variable_number, variables = self._create_exclusive_clauses(var_states)
        # after caling the function the running_variable_number is already increased
        for clause in exclusive_clauses:
            cnf.append(clause)
        ### variable state to parameter ###
        for state, parameter in ds.items():
            # if a parameter is 0 the state is not important
            if parameter == 0:
                continue
            disjunction = []
            # create a disjunction with all states for that parameter
            for var, s in state:
                disjunction.append(-variables[(var, s)])
            # add the parameter
            disjunction.append(running_variable_number)
            # TODO: check if the parameter value is already available
            parameter_vars[running_variable_number] = (parameter, state)
            # add it to the cnf
            cnf.append(disjunction)
            # if the disjunction is not empty
            if len(disjunction) > 2:
                # remove one state by one and add it to the final cnf
                for var, s in state:
                    tmp_disjunction = disjunction.copy()
                    if -variables[(var, s)] in tmp_disjunction:
                        tmp_disjunction.remove(-variables[(var, s)])
                    for i, v in enumerate(tmp_disjunction):
                        tmp_disjunction[i] = -v
                    cnf.append(tmp_disjunction)
            # one discrete variable special case
            elif len(state) == 1:
                tmp_disjunction = disjunction.copy()
                for i, v in enumerate(tmp_disjunction):
                    tmp_disjunction[i] = -v
                cnf.append(tmp_disjunction)
            running_variable_number += 1

        return cnf, parameter_vars, variables

    def get_cnf(self, interactions, verbose=False):
        # create the logical formula as string
        formula, parameter_vars, variables = self._construct_cnf(interactions, verbose)
        self.inst_to_number = {value: key for key, value in variables.items()}
        self.par_values = parameter_vars
        return formula

    def _build_data_structure(self, q, interactions=2, all=False):
        """
        Transforms the parameter tensor into a new data structure, which contains a dictionary with the states
        and the parameter for each variable combination
        :param q: parameter tensor
        :return: transformed data structure, where each key is a valid state
            and each value is the parameter describing this state
        """
        shape = np.shape(q)
        n_vars = shape[0]
        # in the shape at point interactions we have the number of states for each variable
        # example for 3 variables and 2 states interactions 2: shape(3,3,2,2)
        # example for 5 variables and 3 states interactions 3: shape(5,5,5,3,3,3)
        var_states = {var: list(range(shape[interactions])) for var in range(n_vars)}
        # a state is now a tuple of variables for each state we sum the parameter that are valid
        states = list(it.product(list(range(n_vars)), list(range(shape[interactions]))))
        # combination of pairwise states (we remove the states, over same variables)
        # ((0,0), (1,0)) is for example the interaction of var 0 value 0 with var 1 value 0
        state_to_parameter = {(i,j): 0.0 for i,j in it.combinations(states, r=interactions) if i[0] != j[0]}
        for state in state_to_parameter.keys():
            # the index list represents the index for a value in the interaction tensor q
            index = []
            # the first values of this index are the variables
            for var, _ in state:
                index.append(var)
            # for each variable and state combination we have to add the univariate parameter as well
            # the second are the states
            for var, value in state:
                index.append(value)
            # interaction parameter for the configured state
            state_to_parameter[state] = q[tuple(index)]
        # remove states with parameter zero
        states_to_remove = []
        for state, value in state_to_parameter.items():
            if value == 0 and not all:
                states_to_remove.append(state)
        for state in states_to_remove:
            state_to_parameter.pop(state, None)
        # calculate the exp(state) of every state:
        for state, value in state_to_parameter.items():
            state_to_parameter[state] = np.exp(value)
        return state_to_parameter, var_states

    '''
    Added by Paul Kahlmeyer:

    For binary bayesian network
    '''

    def _build_data_structure_bayesnet(self, prob_tables):
        n = len(prob_tables)
        state2param = {}
        var_states = {var: [0,1] for var in range(n)}

        # iterate over each node: 
        for i in range(n):
            sample_space, probs, idx = prob_tables[i]

            # iterate over each case
            for s, prob in zip(sample_space, probs):
                event = tuple([(state, status) for state, status in zip(idx, s)])
                state2param[event] = prob
        return state2param, var_states

    def _construct_cnf_bayesnet(self, prob_tables, verbose=False):
        parameter_vars = {}
        cnf = []
        ds, var_states = self._build_data_structure_bayesnet(prob_tables)

        ### exclusive variable states ###
        exclusive_clauses, running_variable_number, variables = self._create_exclusive_clauses(var_states)
        # after caling the function the running_variable_number is already increased
        for clause in exclusive_clauses:
            cnf.append(clause)
        ### variable state to parameter ###
        for state, parameter in ds.items():
            # if a parameter is 0 the state is not important
            if parameter == 0:
                continue
            disjunction = []
            # create a disjunction with all states for that parameter
            for var, s in state:
                disjunction.append(-variables[(var, s)])
            # add the parameter
            disjunction.append(running_variable_number)
            # TODO: check if the parameter value is already available
            parameter_vars[running_variable_number] = (parameter, state)
            # add it to the cnf
            cnf.append(disjunction)
            # if the disjunction is not empty
            if len(disjunction) > 2:
                # remove one state by one and add it to the final cnf
                for var, s in state:
                    tmp_disjunction = disjunction.copy()
                    if -variables[(var, s)] in tmp_disjunction:
                        tmp_disjunction.remove(-variables[(var, s)])
                    for i, v in enumerate(tmp_disjunction):
                        tmp_disjunction[i] = -v
                    cnf.append(tmp_disjunction)
            # one discrete variable special case
            elif len(state) == 1:
                tmp_disjunction = disjunction.copy()
                for i, v in enumerate(tmp_disjunction):
                    tmp_disjunction[i] = -v
                cnf.append(tmp_disjunction)
            running_variable_number += 1

        return cnf, parameter_vars, variables

    def get_cnf_bayesnet(self, prob_tables, verbose=False):
        # create the logical formula as string
        formula, parameter_vars, variables = self._construct_cnf_bayesnet(prob_tables, verbose)
        self.inst_to_number = {value: key for key, value in variables.items()}
        self.par_values = parameter_vars
        return formula


class CNF(object):
    def __init__(self, cnf):
        self.cnf = cnf

    def _get_vars(self):
        var_list = [element for clause in self.cnf for element in clause]
        return var_list

    def _count_variables(self):
        var_list = self._get_vars()
        assert all([isinstance(element, Number) for element in var_list])
        # remove negations
        var_list = np.abs(var_list)
        # cound different numbers
        n_vars = len(np.unique(var_list))
        return n_vars

    def _get_eclauses(self, num_to_var):
        """
        Creates the eclauses used by the compiler
        :param num_to_var: number of the cnf clause to a variable
        :return:
        """
        n_vars = self._count_variables()
        eclauses = [[] for i in range(n_vars)]
        for variable in num_to_var.keys():
            if variable.startswith("x"):
                eclauses[int(variable[1])].append(num_to_var[variable])
        return [clause for clause in eclauses if clause]

    def create_cnf_file(self, filename="graphical_model.cnf", num_to_var=None):
        """
        Creates the cnf file for the compiler
        :param file_name: name of the cnf file
        :return: file name of the cnf file
        """
        n_vars = self._count_variables()
        n_clauses = len(self.cnf)
        if num_to_var:
            pass
            # eclauses = self._get_eclauses(num_to_var)
        ensure_path_exists(DIR_NNF)
        cnf_file = Path(f"{DIR_NNF}/{filename}")
        with open(cnf_file, "w") as file:
            # p cnf number_vars number_clauses
            file.write(f"p cnf {n_vars} {n_clauses}\n")
            for clause in self.cnf:
                # 0 ends the line
                file.write(f"{' '.join([str(element) for element in clause])} 0\n")
        return filename

    def _replace_vars(self):
        var_list = self._get_vars()
        raise NotImplementedError("Vars are only allowed to be numbers.")

    def print_cnf(self):
        # combine the elements with or
        normal_form = [" or ".join([str(element) for element in clause]) for clause in self.cnf]
        # insert brackets
        normal_form = [f"({i})" for i in normal_form]
        # combine the clauses with and
        normal_form = " and ".join(normal_form)
        print(normal_form)

    def get_cnf(self):
        return self.cnf

    def set_cnf(self, cnf):
        self.cnf = cnf

class CnfToExpTree(object):
    def __init__(self, miniC2D=True):
        self.miniC2D = miniC2D
        if platform.system() == "Windows":
            self.compiler_path = Path("compiler/util/c2d", "c2d_win.exe")
        elif platform.system() == "Linux":
            if miniC2D:
                self.compiler_path = Path("compiler/util/c2d", "miniC2D")
            else:
                self.compiler_path = Path("compiler/util/c2d", "c2d_linux")
        else:
            raise Exception("Platform is not supported")
        if not self.compiler_path.exists():
            raise Exception(f"Compiler files not found, check path '{self.compiler_path}'")

    def compile(self, cnf, simplify=True, num_to_var=None, filename="graphical_model.cnf"):
        if isinstance(cnf, str):
            cnf = CNF(cnf)
        cnf_path = cnf.create_cnf_file(num_to_var=num_to_var, filename=filename)
        cnf_file = Path(f"{DIR_NNF}/{cnf_path}")
        if not cnf_file.exists():
            raise Exception("Please create a cnf file first using 'create_cnf_file'.")
        try:
            if self.miniC2D and platform.system() == "Linux":
                command = [str(self.compiler_path), "-c", str(cnf_file)]
            else:
                command = [str(self.compiler_path), "-in", str(cnf_file)]
            if simplify and not self.miniC2D:
                command.append("-reduce")
            output = subprocess.check_output(command)
        except subprocess.CalledProcessError as cpe:
            print(f"Error during compilation process: {cpe.stdout.decode('utf-8')}")
            raise
        # return the graph
        try:
            return nnf_file_to_graph(cnf_path)
        except:
            raise

class ExprTree(object):
    def __init__(self):
        self.nnf_root = None
        self.n_nodes = 0

    @LM.logfunc
    def create_ET(self, number_to_instance, parameter_values, q, id_to_var_ids):
        # Working list containing the current node and the parent
        stack = deque([(self.nnf_root, None)])
        root = None
        while stack:
            node, parent = stack.popleft()
            # create the node according to the state
            if node.label == "U":
                p = Node_internal(ENT.ETYPE_ADD, parent=parent)
            elif node.label == "M":
                p = Node_internal(ENT.ETYPE_MUL, parent=parent)
            elif node.label == "V":
                (var, state) = node.var
                # case indicator variable:
                p = Node_internal(expr_type=ENT.ETYPE_MUL, parent=parent)
                input_indicator_i = Node_leaf_input(parent=p, var_id=var, cat_value=state)
                # add univariate potentials:
                tmp = q[id_to_var_ids[var], id_to_var_ids[var], state, state]
                if not tmp == 0:
                    #param_q_uni = Node_leaf_param(parent=p, theta=np.exp(0.5*tmp))
                    param_q_uni = Node_leaf_param(parent=p, theta=tmp)
            elif node.label.startswith("P"):
                if isinstance(node.var, float):
                    p = Node_leaf_param(theta=node.var, parent=parent)

            elif node.label == "0":
                pass
            else:
                print(node)
                raise Exception("Error during transformation into ET. Node type not known.")
            # if a node has no parent it is the root
            if not parent:
                root = p
            # continue with the children of the node
            for child in node.get_children():
                stack.append((child, p))
        return root


    def create(self, root_nnf_node, instance_to_number, parameter_values, var_ids=None):
        # self.nnf_root = root_nnf_node.copy()
        self.nnf_root = root_nnf_node
        # self.n_nodes = max([n.id for n in root_nnf_node.get_descendants()])
        self.n_nodes = root_nnf_node.id
        number_to_instance = instance_to_number
        for node in self.nnf_root._get_nodes():
            if node.id > self.n_nodes:
                self.n_nodes = node.id
            if node.label == "A":
                node.var = "*"
                node.label = "M"
            elif node.label == "O":
                node.var = "+"
                node.label = "U"
            # we just replace the literals
            if node.label == "L":
                var_label = int(np.abs(node.var))
                if var_label in parameter_values.keys():
                    node.label = "P"
                    if node.var < 0:
                        node.label = "0"
                        node.var = 1
                    else:
                        node.label += f"({parameter_values[var_label][1]})"
                        node.var = parameter_values[var_label][0]
                elif var_label in number_to_instance.keys():
                    label = number_to_instance[var_label]
                    node.label = "V"
                    if node.var < 0:
                        node.var = 1
                        node.label = "0"
                    else:
                        if not var_ids:
                            node.var = label
                        else:
                            new_label = (var_ids[label[0]], label[1])
                            node.var = new_label
                else:
                    raise Exception("Literal not known.")


    def simplify(self):
        def _simplify(node):
            to_remove = []
            for child in node.get_children():
                if child.label == "O":
                    to_remove.append(child)
                else:
                    _simplify(child)
            for element in to_remove:
                node.children.remove(element)
        _simplify(self.nnf_root)

    def print_expression(self):
        def _pp(node):
            if node.children:
                if node.var == "+":
                    return f"{node.var}".join([f"({_pp(child)})" for child in node.children])
                else:
                    return f"{node.var}".join([f"{_pp(child)}" for child in node.children])
            else:
                if isinstance(node.var, Number):
                    #return f"{np.round(node.var, 4)}"
                    return f"{node.var}"
                else:
                    return f"{node.var}"
        if self.nnf_root:
            print(_pp(self.nnf_root))
        else:
            print("There is no expression tree.")

    def eval(self, cond):
        """
        Returns the value of the network given the conditions cond
        :param cond: dictionary of tuples (variable, value)
        :return: evaluated value
        """
        # self.simplify()
        def _eval(node):
            if node.label == "M":
                return np.prod([_eval(child) for child in node.get_children()])
            elif node.label == "A":
                return np.sum([_eval(child) for child in node.get_children()])
            elif node.label == "V":
                (var, value) = node.var
                if var in cond.keys():
                    if cond[var] == value:
                        return 1
                    else:
                        return 0
                else:
                    # if the variable is not in the condition we have to marginalize this variable
                    return 1
            elif node.label.startswith("P"):
                return node.var
        if self.nnf_root:
            return _eval(self.nnf_root)
        else:
            raise Warning("You did not create a expression graph.")

class NNFNode(object):
    def __init__(self, label, var, id):
        self.label = label
        self.var = var
        self.children = deque()
        self.id = id

    """
    def get_descendants(self):
        descendats = deque([self])
        for child in self.get_children():
            descendats.append(child)
            descendats += child.get_descendants()
        return descendats
    """

    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_children(self):
        return self.children

    def eval(self, cond):
        raise NotImplementedError("Eval will be available later.")

    def __repr__(self):
        return f"{self.label}: {self.var}"

    def _get_nodes(self):
        nodes = []
        stack = deque([self])
        while stack:
            curr = stack.popleft()
            nodes.append(curr)
            for child in curr.get_children():
                stack.append(child)
        return nodes


    def copy(self):
        tmp = NNFNode(self.label, self.var, self.id)
        tmp.children = [node.copy() for node in self.get_children()]
        return tmp
