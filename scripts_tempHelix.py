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

def direct_sum(A, B):
    # Check if both matrices A and B are two-dimensional
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both input arrays should be two-dimensional")
        
    # Create zero matrices that match the dimensions of the other input
    zero_A = np.zeros((A.shape[0], B.shape[1]))
    zero_B = np.zeros((B.shape[0], A.shape[1]))
    
    # Concatenate these matrices appropriately to get the direct sum
    top = np.hstack([A, zero_A]) # top part of the direct sum
    bottom = np.hstack([zero_B, B]) # bottom part of the direct sum
    
    return np.vstack([top, bottom]) # final direct sum
    
def generate_triangular_graph(i):
    G = nx.Graph()
    G.add_edge(2*i, 2*i+2)
    G.add_edge(2*i+1, 2*i+2)
    return G

def generate_reverse_triangular_graph(i):
    G = nx.Graph()
    # Add edges based on the repetition count
    G.add_edge(3*i+2, 3*i+3)
    G.add_edge(3*i+2, 3*i+4)
    return G

def rotation_matrix():
    # Define the permutation matrix
    P = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    return P
    
def direct_sum_triangular_graph(repetitions):
    result_matrix = np.zeros((2,2))
    for i in range(repetitions):
        G = generate_triangular_graph(i)
        nodes = sorted(G.nodes())
        adjacency_matrix = nx.adjacency_matrix(G, nodelist=nodes).todense()        
        if i == 0:
            result_matrix = adjacency_matrix
        else:
            # Direct sum
            result_matrix = direct_sum(result_matrix, adjacency_matrix)

    return result_matrix

def direct_sum_reverse_triangular_graph(repetitions):
    # Initialize the result matrix to represent two disconnected nodes 0 and 1
    result_matrix = np.zeros((2,2))

    for i in range(repetitions-1):
        G = generate_reverse_triangular_graph(i)
        nodes = sorted(G.nodes())
        adjacency_matrix = nx.adjacency_matrix(G, nodelist=nodes).todense()
        
        # Direct sum
        result_matrix = direct_sum(result_matrix, adjacency_matrix)
    
    # After all repetitions, add an additional disconnected node
    result_matrix = direct_sum(result_matrix, np.zeros((1,1)))
    return result_matrix

def direct_sum_rotation_matrix(repetitions):
    p = rotation_matrix()
    result_matrix = p
    for i in range(repetitions-1):
        p = rotation_matrix()
        result_matrix = direct_sum(result_matrix, p)

    return result_matrix
    
def add_new_vertex(adj_matrix, new_vertex_at_start, connect_to_vertices):
    """
    Adds a new vertex to the adjacency matrix at the beginning or the end and connects it to the specified vertices.
    """
    num_vertices = adj_matrix.shape[0] 
    # Create an extended adjacency matrix
    extended_adj_matrix = np.zeros((num_vertices + 1, num_vertices + 1))
    # Copy the original adjacency matrix into the extended one
    if new_vertex_at_start:
        extended_adj_matrix[1:, 1:] = adj_matrix
    else:
        extended_adj_matrix[:-1, :-1] = adj_matrix
    # Add connections from the new vertex to the specified vertices
    if new_vertex_at_start:
        for vertex in connect_to_vertices:
            extended_adj_matrix[0, vertex + 1] = extended_adj_matrix[vertex + 1, 0] = 1
    else:
        for vertex in connect_to_vertices:
            extended_adj_matrix[-1, vertex] = extended_adj_matrix[vertex, -1] = 1

    return extended_adj_matrix

def extend_rotation_matrix(rot_matrix, new_vertex_at_start):
    """
    Extends the rotation matrix with an additional row and column populated with an identity matrix,
    thereby ensuring that the added vertex is not rotated.
    """
    num_vertices = rot_matrix.shape[0]
    # Create an extended rotation matrix
    extended_rot_matrix = np.zeros((num_vertices + 1, num_vertices + 1))
    # Copy the original rotation matrix into the extended one
    if new_vertex_at_start:
        extended_rot_matrix[1:, 1:] = rot_matrix
        # Add identity element to the top-left corner of the extended rotation matrix
        extended_rot_matrix[0, 0] = 1
    else:
        extended_rot_matrix[:-1, :-1] = rot_matrix
        # Add identity element to the bottom-right corner of the extended rotation matrix
        extended_rot_matrix[-1, -1] = 1

    return extended_rot_matrix

def add_self_loops(adj_matrix):
    """
    Adds self-loops to each vertex in the adjacency matrix according to the number of its neighbors.
    """
    num_vertices = adj_matrix.shape[0]
    
    # Compute degree (number of neighbors) for each vertex
    degree = np.sum(adj_matrix, axis=1)

    # Add self-loops according to the degree
    for vertex in range(num_vertices):
        adj_matrix[vertex, vertex] = degree[vertex]

    return adj_matrix


def rotate_triangular(adjm,rotMat):
    rotated_matrix = np.dot(rotMat, np.dot(adjm, rotMat.T))
    return rotated_matrix

def generate_temporal_helix(repetitions, timeSteps):
    triangularGraphMatrix = direct_sum_triangular_graph(repetitions=repetitions)
    reverseTriangularGraphMatrix = direct_sum_reverse_triangular_graph(repetitions=repetitions)

    graphMatrix = triangularGraphMatrix + reverseTriangularGraphMatrix
    rotationMatrix = direct_sum_rotation_matrix(repetitions)

    # Add L , c0 and R nodes.
    graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = True, connect_to_vertices=[0, 1])
    graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = True, connect_to_vertices=[0])
    graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = False,connect_to_vertices=[-2])

    # Extend the rotation matrix to accommodate the new vertices
    rotationMatrix = extend_rotation_matrix(rotationMatrix, new_vertex_at_start=True)
    rotationMatrix = extend_rotation_matrix(rotationMatrix, new_vertex_at_start=True)
    rotationMatrix = extend_rotation_matrix(rotationMatrix, new_vertex_at_start=False)

    result_matrix = graphMatrix
    for _ in range(timeSteps):
        result_matrix = rotate_triangular(result_matrix, rotationMatrix)
    
    result_matrix = add_self_loops(result_matrix)
    return result_matrix
   
def generate_static_temporal_helix(repetitions):
    triangularGraphMatrix = direct_sum_triangular_graph(repetitions=repetitions)
    reverseTriangularGraphMatrix = direct_sum_reverse_triangular_graph(repetitions=repetitions)

    graphMatrix = triangularGraphMatrix + reverseTriangularGraphMatrix
    rotationMatrix = direct_sum_rotation_matrix(repetitions)

    # Add L , c0 and R nodes.
    graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = True, connect_to_vertices=[0, 1])
    graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = True, connect_to_vertices=[0])
    graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = False,connect_to_vertices=[-2])
    
    result_matrix = add_self_loops(graphMatrix)
    return result_matrix
  
def exponential_temporal_helix(rep,epsilon):
    n = (3+rep*3)/2
    copies = math.floor(n**(1-epsilon))
    graph0_2n = []
    graph1_2n = []    
    graph2_2n = []
    print(f'number of reps: {rep} \tnumber of copies: {copies}')
    for i in range(1,copies+1):
        graph0_2n.append(generate_temporal_helix(rep, 0))
        graph1_2n.append(generate_temporal_helix(rep, 1))
        graph2_2n.append(generate_temporal_helix(rep, 2))
            
    return graph0_2n,graph1_2n,graph2_2n


def multiple_exponential_temporal_helix(repetitions, epsilon):
    graphList = []
    for rep in repetitions:
        graph0,graph1,graph2 = exponential_temporal_helix(rep,epsilon)
        graphList.append(graph0+graph1+graph2)
    return graphList
