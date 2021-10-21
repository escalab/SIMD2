import sys
from random import randint
import random

if len(sys.argv) < 2: 
    print('usage: num verts')
    exit(0)

v = int(sys.argv[1])
density = float(sys.argv[2])

adj_mat = [[0 for _ in range(v)] for _ in range(v)]
num_edges = 0
for i in range(v):
    for j in range(i,v):
        if random.uniform(0,1) < density:
            adj_mat[i][j] = random.uniform(0.1,30)
            adj_mat[j][i] = adj_mat[i][j]
            num_edges += 1
        else:
            adj_mat[i][j] = float('inf')


fp = open(str(v)+'_'+sys.argv[2]+'_dense_pbbs.txt', 'w')

fp.write('WeightedEdgeArray')

edge_dict = set({})
for i in range(v):
    for j in range(i,v):
        if adj_mat[i][j] < float('inf'):
            fp.write('\n'+str(i) + ' ' + str(j) + ' ' + str(adj_mat[i][j]))
fp.close()