import random
import networkx as nx
from matplotlib import pyplot as plt
from collections import Counter
import math
from utils.plotTools import plot_qwak
import os
import ast
import numpy as np
import json
from sklearn.linear_model import LinearRegression

from scripts import load_list_from_file, write_list_to_file, load_or_generate_data, draw_graph, draw_graph_from_adjacency_matrix

def create_transition_matrix(G):
    """
    Create a Markovian transition matrix from a NetworkX graph.

    Parameters:
    G (networkx.classes.graph.Graph): A NetworkX graph.

    Returns:
    numpy.ndarray: The Markovian transition matrix.
    """
    
    # Get the adjacency matrix (A) from the graph
    A = nx.to_numpy_array(G, dtype=np.float64)
    
    # Get the sum of each row in the adjacency matrix
    row_sums = A.sum(axis=1)
    
    # Create the transition matrix (T) by dividing each row in A by the corresponding row sum
    T = np.zeros_like(A)  # Create an empty matrix with the same shape as A
    for i in range(A.shape[0]):
        T[i, :] = A[i, :] / row_sums[i]
    
    return T

def expected_hitting_time(P, q, z):
    """
    Calculate the expected hitting time to state z.

    Parameters:
    P (numpy.ndarray): Transition matrix.
    q (numpy.ndarray): Initial state distribution.
    z (int): Target state.

    Returns:
    float: Expected hitting time to state z from state q.
    """
    
    n = P.shape[0]  # Number of states

    # Create the modified transition matrix P_{-z}
    P_minus_z = P.copy()
    P_minus_z[:, z] = 0  # zero the column corresponding to z
    P_minus_z[z, :] = 0  # zero the row corresponding to z

    # Create the modified initial state distribution q_{-z}
    q_minus_z = q.copy()
    q_minus_z[z] = 0  # zero the element corresponding to z

    # Create the identity matrix
    I = np.eye(n)

    # Calculate the inverse of (I - P_{-z})
    inv = np.linalg.inv(I - P_minus_z)

    # Calculate the expected hitting time from state q to state z
    h_z_q = np.dot(q_minus_z, np.dot(inv, np.ones(n)))

    return h_z_q


