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
import cupy as cp


from scripts import load_list_from_file, write_list_to_file, load_or_generate_data, draw_graph, draw_graph_from_adjacency_matrix

def direct_sum(A, B):
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both input arrays should be two-dimensional")
        
    zero_A = cp.zeros((A.shape[0], B.shape[1]))
    zero_B = cp.zeros((B.shape[0], A.shape[1]))
    
    top = cp.hstack([A, zero_A])
    bottom = cp.hstack([zero_B, B])
    
    return cp.vstack([top, bottom])

def generate_triangular_graph(i):
    # Step 1: Create a Node List
    nodes = [2*i, 2*i+1, 2*i+2]
    
    # Step 2: Initialize the Adjacency Matrix
    n_nodes = 3*i + 3
    adjacency_matrix = cp.zeros((n_nodes, n_nodes), dtype=int)
    
    # Step 3: Add Edges
    adjacency_matrix[2*i, 2*i+2] = 1
    adjacency_matrix[2*i+2, 2*i] = 1
    adjacency_matrix[2*i+1, 2*i+2] = 1
    adjacency_matrix[2*i+2, 2*i+1] = 1
    
    # Step 4: Reorder Rows and Columns
    ordered_matrix = adjacency_matrix[nodes, :][:, nodes]
    
    return ordered_matrix

def generate_reverse_triangular_graph(i):
    # Step 1: Create a Node List
    nodes = [3*i, 3*i+1, 3*i+2]
    
    # Step 2: Initialize the Adjacency Matrix
    num_nodes = 3 * i + 3
    adj_matrix = cp.zeros((num_nodes, num_nodes), dtype=cp.int32)
    
    # Step 3: Add Edges
    adj_matrix[3 * i, 3 * i + 1] = adj_matrix[3 * i + 1, 3 * i] = 1
    adj_matrix[3 * i, 3 * i + 2] = adj_matrix[3 * i + 2, 3 * i] = 1
    
    # Step 4: Reorder Rows and Columns
    ordered_matrix = adj_matrix[nodes, :][:, nodes]
    
    return ordered_matrix

def direct_sum_triangular_graph_cupy(repetitions):
    result_matrix = cp.zeros((2,2))
    for i in range(repetitions):
        adjacency_matrix = generate_triangular_graph(i)     
        if i == 0:
            result_matrix = adjacency_matrix
        else:
            # Direct sum
            result_matrix = direct_sum(result_matrix, adjacency_matrix)
    return result_matrix


def direct_sum_reverse_triangular_graph_cupy(repetitions):
    # Initialize the result matrix to represent two disconnected nodes 0 and 1
    result_matrix = cp.zeros((2,2))

    for i in range(repetitions-1):
        adjacency_matrix = generate_reverse_triangular_graph(i)     
        # Direct sum
        result_matrix = direct_sum(result_matrix, adjacency_matrix)
    
    # After all repetitions, add an additional disconnected node
    result_matrix = direct_sum(result_matrix, cp.zeros((1,1)))
    return result_matrix

def rotation_matrix():
    # Define the permutation matrix
    P = cp.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    return P

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
    extended_adj_matrix = cp.zeros((num_vertices + 1, num_vertices + 1))
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
    extended_rot_matrix = cp.zeros((num_vertices + 1, num_vertices + 1))
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
    degree = cp.sum(adj_matrix, axis=1)

    # Add self-loops according to the degree
    for vertex in range(num_vertices):
        adj_matrix[vertex, vertex] = degree[vertex]

    return adj_matrix

def rotate_triangular(adjm,rotMat):
    rotated_matrix = cp.dot(rotMat, cp.dot(adjm, rotMat.T))
    return rotated_matrix

def generate_temporal_helix_cupy(repetitions, timeSteps):
    triangularGraphMatrix = direct_sum_triangular_graph_cupy(repetitions=repetitions)
    reverseTriangularGraphMatrix = direct_sum_reverse_triangular_graph_cupy(repetitions=repetitions)

    graphMatrix = cp.add(triangularGraphMatrix,reverseTriangularGraphMatrix)
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
   
# def generate_static_temporal_helix(repetitions):
#     triangularGraphMatrix = direct_sum_triangular_graph(repetitions=repetitions)
#     reverseTriangularGraphMatrix = direct_sum_reverse_triangular_graph(repetitions=repetitions)

#     graphMatrix = triangularGraphMatrix + reverseTriangularGraphMatrix
#     rotationMatrix = direct_sum_rotation_matrix(repetitions)

#     # Add L , c0 and R nodes.
#     graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = True, connect_to_vertices=[0, 1])
#     graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = True, connect_to_vertices=[0])
#     graphMatrix = add_new_vertex(graphMatrix, new_vertex_at_start = False,connect_to_vertices=[-2])
    
#     result_matrix = add_self_loops(graphMatrix)
#     return result_matrix
  
# def exponential_temporal_helix(rep,epsilon):
#     n = (3+rep*3)/2
#     copies = math.floor(n**(1-epsilon))
#     graph0_2n = []
#     graph1_2n = []    
#     graph2_2n = []
#     print(f'number of reps: {rep} \tnumber of copies: {copies}')
#     for i in range(1,copies+1):
#         graph0_2n.append(generate_temporal_helix(rep, 0))
#         graph1_2n.append(generate_temporal_helix(rep, 1))
#         graph2_2n.append(generate_temporal_helix(rep, 2))
            
#     return graph0_2n,graph1_2n,graph2_2n


# def multiple_exponential_temporal_helix(repetitions, epsilon):
#     graphList = []
#     for rep in repetitions:
#         graph0,graph1,graph2 = exponential_temporal_helix(rep,epsilon)
#         graphList.append(graph0+graph1+graph2)
#     return graphList
