import sys
from random import randint
import random

if len(sys.argv) < 2: 
    print('usage: num verts')
    exit(0)

num_verts = int(sys.argv[1])
density = float(sys.argv[2])
num_edges = int(num_verts*num_verts*density)

fp = open(str(num_verts)+'_'+sys.argv[2]+'_mcp.txt', 'w')

fp.write(str(num_verts)+'\n')
fp.write(str(num_edges)+'\n')

edge_dict = set({})
for _ in range(num_edges):
    vert_pair = (randint(0,num_verts-1),randint(0,num_verts-1))
    while vert_pair in edge_dict or vert_pair[0] == vert_pair[1] or (vert_pair[1],vert_pair[0]) in edge_dict:
         vert_pair = (randint(0,num_verts-1),randint(0,num_verts-1))
    edge_dict.add(vert_pair)
    fp.write('\n'+str(vert_pair[0]) + ' ' + str(vert_pair[1]) + ' ' + str(random.uniform(0.1, 30)))

fp.close()

