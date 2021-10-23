import sys
from random import randint
import random

if len(sys.argv) < 2: 
    print('usage: num verts')
    exit(0)

v = int(sys.argv[1])
density = float(sys.argv[2])


# adj_mat = [[0 for _ in range(v)] for _ in range(v)]

print('write to file')
fp = open(str(v)+'_'+sys.argv[2]+'_dense_pbbs.txt', 'w')
fp.write('WeightedEdgeArray')
for i in range(v):
    for j in range(i,v):
        if random.uniform(0,1) < density:
            w = random.uniform(0.1,100)
            fp.write('\n'+str(i) + ' ' + str(j) + ' ' + str(w))
            fp.write('\n'+str(j) + ' ' + str(i) + ' ' + str(w)) 
fp.close()