
# coding: utf-8



import networkx as nx
from collections import Counter
import random
import math
import matplotlib.pyplot as plt




#Add single edge
def insert_edge(G, n1, n2):
    if n1 == n2:
        raise AssertionError("Cannot add self-loop")
    if n1 < n2:
        G.add_edge(n1, n2)
    else:
        G.add_edge(n2, n1)
    return G




#Add edges
def insert_edges(G, edges):
    for edge in edges:
        G = insert_edge(G, edge[0], edge[1])
    return G




#Check if the core values given are actually realizable
def is_realizable(cores):
    cores = sorted(cores, reverse=True)
    num_max_core_val = max(idx for idx, val in enumerate(cores) if val == cores[0])
    if num_max_core_val < cores[0]:
        return False
    return True




#Add lower core value nodes
def add_lower_nodes(cores, indices, G, k):
    K = cores[0]
    higher_nodes = [node for i in range(k+1, K+1) for node in indices[i]]
    G.add_nodes_from(indices[k])
    for node in indices[k]:
        rands = random.sample(higher_nodes, k)
        G = insert_edges(G, [(r, node) for r in rands])
    if k == 0:
        return G
    return add_lower_nodes(cores, indices, G, k-1)




#Build up single core graph
def generate_k_graph(C, N, G):
    if not C%2 == N%2:
        G.add_nodes_from(range(N))
        G = insert_edges(G, [(i, i+1) for i in range(N-1)])
        G = insert_edge(G, 0, N-1)
        z = math.ceil((N-C+1)/2)
        print(z)
        for i in range(N):
            print(i)
            start = (i+z+1)%(N)
            stop = (i-z)%(N)
            print(start)
            print(stop)
            print()
            if stop >= start:
                edges = set([r for r in range(list(range(N))[(start)],  list(range(N))[(stop)])])
            else:
                edges = set([r for l in [range(list(range(N))[start], N), range(0, list(range(N))[stop])] for r in l])
            G = insert_edges(G, [(i, e) for e in edges])
    else:
        G = generate_k_graph(C, N-1, G)
        G.add_node(N-1)
        rands = random.sample(range(N-1), C)
        G = insert_edges(G, [(N-1, r) for r in rands])
    return G



#Main function: give a list of core values
def generate_graph(cores):
    cores = sorted(cores, reverse=True)
    if not is_realizable(cores):
        return False
    K = cores[0]
    nums = Counter(cores)
    indices = {i : [idx for idx, val in enumerate(cores) if val == i] for i in range(0, K+1)}
    G = nx.Graph()
    if K == 0:
        G.add_nodes_from(indices[0])
        return G
    if K == 1:
        G.add_nodes_from(indices[1])
        for i in range(nums[1]-1):
            G = insert_edge(G, i, i+1)
        G.add_nodes_from(indices[0])
        return G
    if K == 2:
        G.add_nodes_from(indices[2])
        for i in range(nums[2]-1):
            G = insert_edge(G, i, i+1)
        G = insert_edge(G, 0, nums[2]-1)
        return add_lower_nodes(cores, indices, G, 1)
    else:
        G = generate_k_graph(K, nums[K], G)
        G = add_lower_nodes(cores, indices, G, K-1)
        return G

