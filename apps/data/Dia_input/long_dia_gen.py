import os
import sys
import random

def main():
    if len(sys.argv) < 2: 
        print('usage: num_verts')
        exit(0)

    num_verts = int(sys.argv[1])
    fp = open(str(num_verts)+'_LD.txt', 'w')
    fp.write(str(num_verts) + ' ')
    fp.write(str(num_verts-1) + '\n')
    for i in range(num_verts-1): 
        fp.write( str(i) + ' ' + str(i+1) + ' ' + str(random.randint(1,10)) + '\n')

    

if __name__ == "__main__":
    main()