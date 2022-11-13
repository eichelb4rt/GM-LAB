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

def main():
    n = 9
    monomials = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1), 5: (2, 0), 6: (2, 1), 7: (3, 0), 8: (3, 1), 9: (4, 0), 10: (4, 1), 11: (5, 0), 12: (5, 1), 13: (6, 0), 14: (6, 1), 15: (7, 0), 16: (7, 1), 17: (8, 0), 18: (8, 1)}
    cond_monomials = {19: (0.7658227848101266, ((7, 0), (0, 0))), 21: (0.20748299319727892, ((7, 1), (0, 0))), 23: (0.692, ((8, 0), (1, 0))), 25: (0.27238805970149255, ((8, 1), (1, 0))), 27: (0.6075949367088608, ((1, 0), (5, 0), (2, 0))), 29: (0.3791208791208791, ((1, 0), (5, 1), (2, 0))), 31: (0.35294117647058826, ((1, 1), (5, 0), (2, 0))), 33: (0.2676056338028169, ((1, 1), (5, 1), (2, 0))), 35: (0.45703125, ((3, 0),)), 37: (0.8571428571428571, ((1, 0), (3, 0), (4, 0))), 39: (0.6111111111111112, ((1, 0), (3, 1), (4, 0))), 41: (0.8040540540540541, ((1, 1), (3, 0), (4, 0))), 43: (0.26865671641791045, ((1, 1), (3, 1), (4, 0))), 45: (0.6894586894586895, ((3, 0), (5, 0))), 47: (0.31414868105515587, ((3, 1), (5, 0))), 49: (0.6784968684759917, ((4, 0), (6, 0))), 51: (0.5121107266435986, ((4, 1), (6, 0))), 53: (0.8577586206896551, ((2, 0), (8, 0), (7, 0))), 55: (0.627906976744186, ((2, 0), (8, 1), (7, 0))), 57: (0.5708955223880597, ((2, 1), (8, 0), (7, 0))), 59: (0.37362637362637363, ((2, 1), (8, 1), (7, 0))), 61: (0.774798927613941, ((5, 0), (8, 0))), 63: (0.5341772151898734, ((5, 1), (8, 0))), 20: (0.23417721518987344, ((7, 0), (0, 1))), 22: (0.7925170068027211, ((7, 1), (0, 1))), 24: (0.30800000000000005, ((8, 0), (1, 1))), 26: (0.7276119402985075, ((8, 1), (1, 1))), 28: (0.3924050632911392, ((1, 0), (5, 0), (2, 1))), 30: (0.6208791208791209, ((1, 0), (5, 1), (2, 1))), 32: (0.6470588235294117, ((1, 1), (5, 0), (2, 1))), 34: (0.7323943661971831, ((1, 1), (5, 1), (2, 1))), 36: (0.54296875, ((3, 1),)), 38: (0.1428571428571429, ((1, 0), (3, 0), (4, 1))), 40: (0.38888888888888884, ((1, 0), (3, 1), (4, 1))), 42: (0.19594594594594594, ((1, 1), (3, 0), (4, 1))), 44: (0.7313432835820896, ((1, 1), (3, 1), (4, 1))), 46: (0.3105413105413105, ((3, 0), (5, 1))), 48: (0.6858513189448441, ((3, 1), (5, 1))), 50: (0.32150313152400833, ((4, 0), (6, 1))), 52: (0.4878892733564014, ((4, 1), (6, 1))), 54: (0.14224137931034486, ((2, 0), (8, 0), (7, 1))), 56: (0.37209302325581395, ((2, 0), (8, 1), (7, 1))), 58: (0.42910447761194026, ((2, 1), (8, 0), (7, 1))), 60: (0.6263736263736264, ((2, 1), (8, 1), (7, 1))), 62: (0.22520107238605902, ((5, 0), (8, 1))), 64: (0.4658227848101266, ((5, 1), (8, 1)))}
    cnf = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [-1, -2], [-3, -4], [-5, -6], [-7, -8], [-9, -10], [-11, -12], [-13, -14], [-15, -16], [-17, -18], [-19, 1], [-21, 1], [-23, 3], [-25, 3], [-27, 5], [-29, 5], [-31, 5], [-33, 5], [-35, 7], [-37, 9], [-39, 9], [-41, 9], [-43, 9], [-45, 11], [-47, 11], [-49, 13], [-51, 13], [-53, 15], [-55, 15], [-57, 15], [-59, 15], [-61, 17], [-63, 17], [-20, 2], [-22, 2], [-24, 4], [-26, 4], [-28, 6], [-30, 6], [-32, 6], [-34, 6], [-36, 8], [-38, 10], [-40, 10], [-42, 10], [-44, 10], [-46, 12], [-48, 12], [-50, 14], [-52, 14], [-54, 16], [-56, 16], [-58, 16], [-60, 16], [-62, 18], [-64, 18], [-19, 15], [-21, 16], [-23, 17], [-25, 18], [-27, 3], [-27, 11], [-29, 3], [-29, 12], [-31, 4], [-31, 11], [-33, 4], [-33, 12], [-37, 3], [-37, 7], [-39, 3], [-39, 8], [-41, 4], [-41, 7], [-43, 4], [-43, 8], [-45, 7], [-47, 8], [-49, 9], [-51, 10], [-53, 5], [-53, 17], [-55, 5], [-55, 18], [-57, 6], [-57, 17], [-59, 6], [-59, 18], [-61, 11], [-63, 12], [-20, 15], [-22, 16], [-24, 17], [-26, 18], [-28, 3], [-28, 11], [-30, 3], [-30, 12], [-32, 4], [-32, 11], [-34, 4], [-34, 12], [-38, 3], [-38, 7], [-40, 3], [-40, 8], [-42, 4], [-42, 7], [-44, 4], [-44, 8], [-46, 7], [-48, 8], [-50, 9], [-52, 10], [-54, 5], [-54, 17], [-56, 5], [-56, 18], [-58, 6], [-58, 17], [-60, 6], [-60, 18], [-62, 11], [-64, 12], [-19, -15], [-21, -16], [-23, -17], [-25, -18], [-27, -3, -11], [-29, -3, -12], [-31, -4, -11], [-33, -4, -12], [-35], [-37, -3, -7], [-39, -3, -8], [-41, -4, -7], [-43, -4, -8], [-45, -7], [-47, -8], [-49, -9], [-51, -10], [-53, -5, -17], [-55, -5, -18], [-57, -6, -17], [-59, -6, -18], [-61, -11], [-63, -12], [-20, -15], [-22, -16], [-24, -17], [-26, -18], [-28, -3, -11], [-30, -3, -12], [-32, -4, -11], [-34, -4, -12], [-36], [-38, -3, -7], [-40, -3, -8], [-42, -4, -7], [-44, -4, -8], [-46, -7], [-48, -8], [-50, -9], [-52, -10], [-54, -5, -17], [-56, -5, -18], [-58, -6, -17], [-60, -6, -18], [-62, -11], [-64, -12]]
    to_circuit(n, monomials, cond_monomials, cnf)
    

if __name__ == "__main__":
    main()