import os
import sys
import random
if len(sys.argv) < 2: 
    print('usage filename')
    exit()
rmat_file = sys.argv[1]
print(rmat_file)
num_v = 0
num_e = 0
fo = open('mst_'+rmat_file,'w')
fo.write('WeightedEdgeArray\n')

with open(rmat_file, 'r') as fi:
    lines = fi.readlines()
    num_v = int(lines[0].split(' ')[0])
    for line in lines[1:]:
        e = line.strip('\n').split('\t')
        s = int(e[0])
        t = int(e[1])
        w = str(random.uniform(0, 100000))
        fo.write(str(s) + ' ' + str(t) + ' ' + w + '\n')

fo.close()

