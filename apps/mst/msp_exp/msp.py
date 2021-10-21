from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import itertools
import random
from random import randint
import sys
import networkx as nx

source_graph = [[0, 8, 0, 3, 1],
                [8, 0, 2, 5, 0],
                [0, 2, 0, 6, 6],
                [3, 5, 6, 0, 0],
                [1, 0, 6, 0, 0]]

# gengraph
num_vert = int(sys.argv[1])
random.seed(10)
source_graph = [[0 for _ in range(num_vert)] for _ in range(num_vert)]
for i in range(num_vert):
    for j in range(i,num_vert):
        if i == j:  source_graph[i][j] = 0
        else:
            if randint(0,10) > 7:
                source_graph[i][j] = randint(1,15)
                # source_graph[j][i] = source_graph[i][j]
            else: 
                source_graph[i][j] = 0
                # source_graph[j][i] = 0
        
source_graph_sym = [[0 for _ in range(num_vert)] for _ in range(num_vert)]
for i in range(num_vert):
    for j in range(i,num_vert):
        source_graph_sym[i][j] = source_graph[i][j]
        source_graph_sym[j][i] = source_graph[i][j]

print('orignal graph:')
print(np.array(source_graph))

# print('sym graph:')
# print(np.array(source_graph_sym))
# use scipy.sparse to get msp of source graph
csr_graph = csr_matrix(source_graph)
msp_graph = minimum_spanning_tree(csr_graph)
msp_graph = msp_graph.toarray().astype(int)

# run proposed algorithm

def min_max_itr(adj_mat):
    ret_mat = adj_mat
    num_vetex = len(adj_mat)
    for i in range(num_vetex):
        for j in range(num_vetex):
            temp_val = float('inf')
            for k in range(num_vetex):
                temp_val = min(temp_val,max(adj_mat[i][k],adj_mat[k][j]))
            ret_mat[i][j] = min(ret_mat[i][j],temp_val)
    return ret_mat


# parse adj graph, 0 -> inf
parsed_adj_mat = [[0 for _ in range(num_vert)] for _ in range(num_vert)] 
for i, j in itertools.product(range(num_vert), range(num_vert)): 
    if source_graph_sym[i][j] == 0 and i != j: parsed_adj_mat[i][j] = float('inf')
    else: parsed_adj_mat[i][j] = source_graph_sym[i][j]
# print('\nparsed graph:')
# print(np.array(parsed_adj_mat))

# run proposed algorithm
prev_res = None
iter_count = 0
while prev_res != parsed_adj_mat:
    iter_count += 1
    prev_res = parsed_adj_mat
    parsed_adj_mat = min_max_itr(parsed_adj_mat)
    # print('\nresult after {} iteration'.format(iter_count))
    # print(np.array(parsed_adj_mat))
    

print(iter_count,'iteration taken')

for i, j in itertools.product(range(len(parsed_adj_mat)), range(len(parsed_adj_mat))): 
    if parsed_adj_mat[i][j] != source_graph[i][j]: 
        parsed_adj_mat[i][j] = 0
print('\nfinal result:')
print(np.array(parsed_adj_mat))

print('expected: ')
print(msp_graph)

# test correctness on upper triangle
print('result weight sum: ',sum(sum(np.array(parsed_adj_mat))))
print('expect weight sum: ',sum(sum(msp_graph)))

# test if identical/# differencce
num_diff = 0
for i in range(num_vert):
    for j in range(i,num_vert):
        if msp_graph[i][j] != parsed_adj_mat[i][j]:
            num_diff += 1
if num_diff > 0: print('{} edge are different'.format(num_diff))
else: print('msp identical')