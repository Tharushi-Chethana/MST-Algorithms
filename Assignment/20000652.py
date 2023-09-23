import networkx as nx
import random
import numpy as np
import time

# Function to measure execution time of an algorithm
def measure_execution_time(algorithm_func, graph, algorithm_name):
    start_time = time.time()
    result = algorithm_func(graph)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{algorithm_name} Execution Time: {execution_time:.6f} seconds")
    return result


# Kruskal's Algorithm
def kruskal(graph):
    mst = []
    edges = list(graph.edges(data=True))
    # Assign weight to the edges
    edges_with_weights = [(u, v, random.randint(1, 10)) for u, v, _ in edges]
    # Sort the weight
    edges_with_weights = sorted(edges_with_weights, key=lambda x: x[2])
    
    disjoint_sets = {node: {node} for node in graph.nodes()}
    # print(disjoint_sets)

    for edge in edges_with_weights:
        u, v, weight = edge
        # check whether exit any loop or parallel edges
        if disjoint_sets[u] != disjoint_sets[v]:
            mst.append((u, v, weight))
            union(disjoint_sets, u, v)
    return mst


def union(disjoint_sets, u, v):
    set_u = disjoint_sets[u]
    set_v = disjoint_sets[v]
    set_u.update(set_v)
    for node in set_v:
        disjoint_sets[node] = set_u
    # print(disjoint_sets)




# Prim's MST Algorithm (Lazy Version)
def prim_lazy(graph):
    mst = []
    visited = set()
    # Get the start node by selecting the first node in the graph
    start_node = list(graph.nodes())[0]
    #get the priority queue
    pq = [(0, None, start_node)]

    while pq:
        # Pop the node with the lowest weight from the priority queue
        weight, _, node = pq.pop(0)
        if node not in visited:
            # Mark the current node as visited
            visited.add(node)
            # If the current node has a parent (not None), add the edge to the MST
            if _ is not None:
                mst.append((_, node, weight))
            # Explore neighbors of the current node
            for neighbor, edge_data in graph[node].items():
                # Default weight to 1 if 'weight' attribute is missing
                edge_weight = edge_data.get('weight', 1)  
                if neighbor not in visited:
                    pq.append((edge_weight, node, neighbor))
    
    return mst


# Prim's MST Algorithm (Eager Version) for Dense or Sparse Graph (Adjacency List or Matrix)
def prim_eager(graph):
    mst = []
    visited = set()
    if isinstance(graph, nx.Graph):
        # For sparse graph represented as an adjacency list
        start_node = list(graph.nodes())[0]
        key = {node: float('inf') for node in graph.nodes()}
        key[start_node] = 0
        # Priority queue containing (key, node)
        pq = [(0, start_node)]
        # Dictionary to store the parent node
        node_from = {node: None for node in graph.nodes()}  
        
        while pq:
            # Get the node with the smallest key from the priority queue
            weight, node = pq.pop(0)  
            if node not in visited:
                visited.add(node)
                if weight > 0:
                    # Add the edge to the MST
                    mst.append((node, node_from[node], weight))  
                for neighbor, edge_data in graph[node].items():
                    if neighbor not in visited and edge_data.get('weight', 1) < key[neighbor]:
                        key[neighbor] = edge_data.get('weight', 1)  # Update the key of the neighbor
                        node_from[neighbor] = node  # Update the parent of the neighbor
                        pq.append((key[neighbor], neighbor))  # Add the neighbor to the priority queue
    
    elif isinstance(graph, np.ndarray):
        # For dense graph represented as an adjacency matrix
        num_nodes = len(graph)
        start_node = 0
        key = {node: float('inf') for node in range(num_nodes)}
        node_from = {node: None for node in range(num_nodes)}
        key[start_node] = 0
        visited = set()

        while len(visited) < num_nodes:
            min_node = None
            min_key = float('inf')

            for node in range(num_nodes):
                if node not in visited and key[node] < min_key:
                    min_node = node
                    min_key = key[node]

            if min_node is None:
                break

            visited.add(min_node)

            if node_from[min_node] is not None:
                mst.append((min_node, node_from[min_node], min_key))  # Add the edge to the MST

            for node in range(num_nodes):
                if node not in visited and graph[min_node][node] < key[node]:
                    key[node] = graph[min_node][node]  # Update the key of the neighbor
                    node_from[node] = min_node  # Update the parent of the neighbor

    return mst

# Generate random dense graph using adjacency matrix
dense_graph_matrix = nx.fast_gnp_random_graph(n=1000, p=0.5)
dense_matrix = nx.to_numpy_array(dense_graph_matrix)
for u, v in dense_graph_matrix.edges():
    # Assign random weights to edges of the dense graph
    dense_graph_matrix[u][v]['weight'] = random.randint(1, 10)


# Generate random sparsegraph using adjacency list
sparse_graph_list = nx.gnm_random_graph(n=1000, m=50)
for u, v in sparse_graph_list.edges(): 
    # Assign random weights to edges
    sparse_graph_list[u][v]['weight'] = random.randint(1, 10)  
 

# Usage examples
dense_mst_kruskal = kruskal(dense_graph_matrix) 
sparse_mst_kruskal = kruskal(sparse_graph_list)

dense_mst_lazy = prim_lazy(dense_graph_matrix)
sparse_mst_lazy = prim_lazy(sparse_graph_list)

dense_mst_eager = prim_eager(dense_graph_matrix)
sparse_mst_eager = prim_eager(sparse_graph_list)


# Measure execution time for Kruskal's algorithm on dense and sparse graphs using adjacency list and adjacency matrix
kruskal_sparse_list = measure_execution_time(kruskal, sparse_graph_list, "Kruskal (Sparse, List)")
# Convert dense_matrix to a NetworkX Graph
dense_graph_networkx = nx.Graph(dense_matrix)
kruskal_sparse_matrix = measure_execution_time(kruskal, nx.to_networkx_graph(dense_graph_networkx), "Kruskal (Sparse, Matrix)")
kruskal_dense_list = measure_execution_time(kruskal, nx.to_networkx_graph(dense_graph_networkx), "Kruskal (Dense, List)")
kruskal_dense_matrix = measure_execution_time(kruskal, nx.to_networkx_graph(dense_graph_networkx), "Kruskal (Dense, Matrix)")


# Measure execution time for Prim's (Lazy) algorithm on dense and sparse graphs using adjacency list and adjacency matrix
prim_lazy_sparse_list = measure_execution_time(prim_lazy, sparse_graph_list, "Prim (Lazy, Sparse, List)")
prim_lazy_sparse_matrix = measure_execution_time(prim_lazy, nx.to_networkx_graph(dense_graph_networkx), "Prim (Lazy, Sparse, Matrix)")
# Continue using dense_graph_networkx for dense graphs
prim_lazy_dense_list = measure_execution_time(prim_lazy, nx.to_networkx_graph(dense_graph_networkx), "Prim (Lazy, Dense, List)")
prim_lazy_dense_matrix = measure_execution_time(prim_lazy, nx.to_networkx_graph(dense_graph_networkx), "Prim (Lazy, Dense, Matrix)")

# Measure execution time for Prim's (Eager) algorithm on dense and sparse graphs using adjacency list and adjacency matrix
prim_eager_sparse_list = measure_execution_time(prim_eager, sparse_graph_list, "Prim (Eager, Sparse, List)")
prim_eager_sparse_matrix = measure_execution_time(prim_eager, dense_matrix, "Prim (Eager, Sparse, Matrix)")
prim_eager_dense_list = measure_execution_time(prim_eager, nx.to_networkx_graph(dense_matrix), "Prim (Eager, Dense, List)")
prim_eager_dense_matrix = measure_execution_time(prim_eager, nx.to_networkx_graph(dense_matrix), "Prim (Eager, Dense, Matrix)")