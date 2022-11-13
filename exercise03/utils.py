'''
Class for handling Bayesian Networks on binary data
'''

import numpy as np
import itertools
import networkx as nx


# compilation to arithmetic circuit

from compiler import compiler
import compiler.ArithmeticCircuit.ArithmeticCircuit as arith_circuit
from compiler.ArithmeticCircuit.Node import Node

def to_circuit(n : int, state_monoms : dict, cond_monoms : dict, cnf_list : list) -> arith_circuit:
    '''
    Converts CNF to arithmetic circuit using the darwiche compiler.

    @Params:
        n...                number of nodes
        state_monoms...     dictionary with key = monomial index, value = tuple (node index, node value)
        cond_monoms...      dictionary with key = monomial index, value = tuple (prob, conditional state tuple)
        cnf_list...         list of disjunction lists. Each list contains monomial indices ("-" means negated in logical context)

    @Returns
        arithmetic circuit
    '''


    Node.node_id = 0

    var_ids = [i for i in range(n)]
    lf = compiler.LogicalFormula()

    instance_to_number = state_monoms
    parameter_values = cond_monoms
    cnf_object = compiler.CNF(cnf_list)
    ctet = compiler.CnfToExpTree()

    # create logical tree out of cnf
    root = ctet.compile(cnf_object, simplify=True, num_to_var=instance_to_number)
    exp_tree = compiler.ExprTree()
    exp_tree.create(root, instance_to_number, parameter_values, var_ids)
    exp_tree.simplify()

    # create circuit out of logical tree
    q = np.zeros((n, n, 2, 2))
    id_to_var_ids = {}
    for i in range(len(var_ids)):
        id_to_var_ids[var_ids[i]] = i
    tree_root = exp_tree.create_ET(instance_to_number, parameter_values, q, id_to_var_ids)
    ac = arith_circuit.ArithmeticCircuit(tree_root)
    ac.simplify()
    return ac

# Inference Queries for categorical

def prior_marginal(prob_table:np.ndarray, I:np.ndarray) -> np.ndarray:
    '''
    Computes the probability table for a subset of the indices.
    
    @Params:
        prob_table... numpy array with columns holding values, last column holding the probabilities
        I... numpy array with indices
    
    @Returns:
        numpy array with columns holding values, last column holding the probabilities for indices in I
    '''
    sample_spaces = [sorted(list(set(prob_table[:,i]))) for i in I]  
    N = np.prod([len(s) for s in sample_spaces])
    marg_table = np.zeros((N, len(I) + 1))
    for i, comb in enumerate(itertools.product(*sample_spaces)):
        mask = np.all((prob_table[:,I] == comb), axis=1)
        p = np.sum(prob_table[mask, -1])
        marg_table[i, :-1] = np.array(comb)
        marg_table[i, -1] = p
    return marg_table

def posterior_marginal(prob_table:np.ndarray, I:np.ndarray, J:np.ndarray, e_J:np.ndarray) -> np.ndarray:
    '''
    Computes the probability table for a subset of the indices given other subset is set to values.
    
    @Params:
        prob_table... numpy array with columns holding values, last column holding the probabilities
        I... numpy array with indices
        J... numpy array with indices
        e_J... numpy array with values for J
    
    @Returns:
        numpy array with columns holding values, last column holding the probabilities for indices in I
    '''
    # adjust
    print(e_J)
    valid_entries = np.all(prob_table[:, J] == e_J, axis=1)
    prob_table_cond = prob_table[valid_entries]
    
    sample_spaces = [sorted(list(set(prob_table[:,i]))) for i in I]  
    N = np.prod([len(s) for s in sample_spaces])
    
    cond_table = np.zeros((N, len(I) + 1))
    for i, comb in enumerate(itertools.product(*sample_spaces)):
        mask = np.all((prob_table_cond[:,I] == comb), axis=1)
        
        if np.sum(mask) == 0:
            p = 0.0
        else:
            p = np.sum(prob_table_cond[mask, -1])
            
        cond_table[i, :-1] = np.array(comb)
        cond_table[i, -1] = p
    cond_table[:, -1] = cond_table[:, -1]/ cond_table[:, -1].sum()
    return cond_table

def prob_of_evidence(prob_table:np.ndarray, I:np.ndarray, e_I: np.ndarray, J:np.ndarray, e_J:np.ndarray) -> float:
    '''
    Computes the probability of I being e_I given J is e_J.
    
    @Params:
        prob_table... numpy array with columns holding values, last column holding the probabilities
        I... numpy array with indices
        e_I... numpy array with values for I
        J... numpy array with indices
        e_J... numpy array with values for J
    
    @Returns:
        probability of I being e_I given J is e_J.
    '''    
    cond_table = posterior_marginal(prob_table, I, J, e_J)
    
    mask = np.all(cond_table[:,:-1] == e_I, axis=1)
    assert mask.sum() == 1
    return cond_table[mask, -1][0]

def most_prob_explanation(prob_table:np.ndarray, J:np.ndarray, e_J:np.ndarray) -> np.ndarray:
    '''
    Computes the most probable x given some evidence
    
    @Params:
        prob_table... numpy array with columns holding values, last column holding the probabilities
        J... numpy array with indices
        e_J... numpy array with values for J
    
    @Returns:
        x that maximizes probability of x given J is set to e_J
    '''
    I = [i for i in range(prob_table.shape[1] - 1) if i not in J]
    
    cond_table = posterior_marginal(prob_table, I, J, e_J)
    idx = np.argmax(cond_table[:, -1])
    e_I = cond_table[idx, :-1]
    
    x = np.zeros(prob_table.shape[1] - 1)
    x[I] = e_I
    x[J] = e_J
    return x

def max_a_posteriori(prob_table:np.ndarray, I:np.ndarray, J:np.ndarray, e_J:np.ndarray) -> np.ndarray:
    '''
    Computes the most probable x given some evidence
    
    @Params:
        prob_table... numpy array with columns holding values, last column holding the probabilities
        I... numpy array with indices
        J... numpy array with indices
        e_J... numpy array with values for J
    
    @Returns:
        x_I that maximizes probability of x given J is set to e_J
    '''
    cond_table = posterior_marginal(prob_table, I, J, e_J)
    idx = np.argmax(cond_table[:, -1])
    e_I = cond_table[idx, :-1]
    
    return e_I