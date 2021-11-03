import os
import sys
import random

def main():
    if len(sys.argv) < 2: 
        print('usage: num_verts density')
        exit(0)

    num_verts = int(sys.argv[1])
    density = float(sys.argv[2])

    edges_per_vert = int(num_verts * density)

    

    adj_list = []

    for i in range(num_verts-1):
        edge_set = set({})
        for _ in range(edges_per_vert):
            es = random.randint(i+1,num_verts-1)
            if es not in edge_set:
                edge_set.add(es)
                adj_list.append(str(i) + ' ' + str(es) + ' ' + str(random.randint(1,10)) + '\n')
            # fp.write(str(i) + ' ' + str(es) + ' ' + str(random.randint(1,10)) + '\n')


    fp = open(str(num_verts)+'_'+str(density)+'DAG.txt', 'w')
    fp.write(str(num_verts) + ' ')
    fp.write(str(len(adj_list)) + '\n')
    for e in adj_list: fp.write(e)
    fp.close()

    
    

if __name__ == "__main__":
    main()