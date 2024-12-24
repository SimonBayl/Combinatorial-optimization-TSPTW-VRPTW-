from concorde import Problem, run_concorde
import pandas as pd
import time
import numpy as np
import random
import math


def force_final_node(distance_matrix : np.array, final_node : int) :
    """ 
    Modify distance matrix so that edge (final_node,0) is the most interesting
    
    Args : 
        - distance_matrix (2D square numpy.array): Distance matrix for the ATSP.
        - final_node (int): Expected final node in optimal tour
    
    Returns :
        - new_distance_matrix (2D square numpy.array): Modified distance matrix for the ATSP.
    """
    n = len(distance_matrix)
    M = distance_matrix.max()    
    new_distance_matrix = distance_matrix.copy()        
    for j in range(1,n-1): 
        new_distance_matrix[final_node,j] = 10*M 
    return new_distance_matrix


def symmetricize(m: np.array):
    """ 
    Jonker-Volgenant method for transforming asymmetric TSP distance matrix of size n in symmetric TSP distance matrix of size 2*n.

    Args:
        - m (2D square numpy.array): Distance matrix for the ATSP.
    
    Returns:
        - m_symm (2D square numpy.array): Distance matrix for the corresponding STSP.
    """

    high_int = 10 * m.max() 
    m_bar = m.copy()

    m_bar = np.array([[elem + high_int for elem in arr] for arr in m_bar] )
    np.fill_diagonal(m_bar, 0)

    u = np.matrix(np.ones(m.shape) * 10*high_int)
    np.fill_diagonal(u, 0)

    m_symm_top = np.concatenate((u, np.transpose(m_bar)), axis=1)
    m_symm_bottom = np.concatenate((m_bar, u), axis=1)

    m_symm = np.concatenate((m_symm_top, m_symm_bottom), axis=0)

    m_symm= m_symm.astype(np.int64)

    return m_symm


def solve_atsp_concorde(m, final_dump, path_concorde) :
    """
    Solving of the ATSP described by dist_matrix using the Concorde solver.

    To retrieve solution for the ATSP from the STSP solution, pick one node out of two. The direction of the atsp tour is determined by the position of node len(m) since we must have 0 -> len(m) in the stsp tour

    Args :
        - m (2D square numpy.array): Distance matrix for the ATSP.
        - final_dump (int): Expected final dump in optimal tour
        - path_concorde (str):  Path to concorde executable (linux only)

    Returns:
        - atsp_tour (list or np.array): List of nodes describing optimal atsp tour for this problem
    """
    n = len(m)

    m_ajusted = force_final_node(m, final_dump)
    m_symm = symmetricize(m_ajusted)
    problem_stsp = Problem.from_matrix(m_symm)

    solution = run_concorde(problem_stsp, path_concorde)
    if solution.tour[1] == n :
        atsp_tour = [solution.tour[2*i] for i in range(n)]
    elif solution.tour[-1] == n:
        atsp_tour = [solution.tour[2*i] for i in range(0,-n,-1)]
    else : 
        raise ValueError("Transformation from stsp to atsp impossible")
    
    return atsp_tour


def get_tour_cost(tour, distance_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i], tour[i+1]]
    cost += distance_matrix[tour[-1], tour[0]]
    return cost

def main_solve(distance_matrix, path_concorde, dump = [], weights = [], Q = None, init_tour = []) :
    """ 
    Main function to compute optimal tour for the atsp problem defined by distance_matrix. If weights and capacity are given, compute by heuristic a tour respecting capacity constraints.

    Args : 
        - distance_matrix (2D square numpy.array) : Distance matrix for
            the problem, size n x n
        - path_concorde (str) : path to a linux executable for concorde
        - dump (list or np.array) : list of dump nodes
        - weights (list or np.array) : list of weight for each node, size n
        - Q (float) : max capacity
        - init_tour (list or np.array) : Initial tour for the problem. Not used
            to compute optimal tour but allows comparison. Size n
    
    
    """

    if len(dump) == 0 :
        dump = [len(distance_matrix) - 1]

    final_dump = dump[-1]
    opt_tour = solve_atsp_concorde(distance_matrix, final_dump, path_concorde)

    if len(opt_tour) != len(distance_matrix):
        raise ValueError("Missing nodes in tour")
    if set(opt_tour) != set(np.arange(len(distance_matrix))) :
        raise ValueError("Not optimal tour")

    if len(init_tour) > 0:
        init_cost = get_tour_cost(init_tour, distance_matrix)
        print(f'Initial cost = {init_cost}')

    TSP_cost = get_tour_cost(opt_tour, distance_matrix)
    print(f"TSP without intermediate dump cost = {TSP_cost}")

    if len(weights) > 0 and Q:
        if (all(isinstance(item, list) for item in weights)
            or all(isinstance(item, np.ndarray) for item in weights)):
            # Biflux case
            for i in dump :
                # Reset dump weights to 0 (instead of negative values)
                weights[i] = [0, 0]
            
        else:
            # Monoflux case
            for i in dump :
                # Reset dump weights to 0 (instead of negative values)
                weights[i] = 0
        
            if len(dump) != np.ceil(sum(weights)/Q):
                print(np.ceil(sum(weights)/Q))
                print(len(dump))
                raise ValueError("Wrong number of intermediate dumps")
        
        if len(dump) >= 2 :
            # Intermediate dump case
            dump_tour, dump_tour_cost = heuristic_dump_insertion(opt_tour, dump, distance_matrix, weights, Q)

            if len(dump_tour) != len(distance_matrix) :
                raise ValueError("Missing nodes in tour with dumps")
            if set(dump_tour) != set(np.arange(len(distance_matrix))) :
                raise ValueError("Not optimal tour with dumps")
            if dump_tour_cost != get_tour_cost(dump_tour,distance_matrix) :
                raise ValueError("Non matching costs after dump")

            print(f'TSP with dump cost = {dump_tour_cost}')
            print("Best tour : \n" ,dump_tour)
            return dump_tour, dump_tour_cost
    
    print("Best tour : \n" ,opt_tour)
    return opt_tour, TSP_cost
