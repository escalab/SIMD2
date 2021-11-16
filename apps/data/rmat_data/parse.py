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
fo = open('parsed_'+rmat_file,'w')
with open(rmat_file, 'r') as fi:
    lines = fi.readlines()
    num_v = int(lines[0].split(' ')[0])
    for line in lines[1:]:
        e = line.strip('\n').split('\t')
        s = int(e[0])
        t = int(e[1])
        w = str(random.uniform(0, 10))
        fo.write(str(s) + ' ' + str(t) + ' ' + w + '\n')
        num_e += 1

fo.close()

# fo = open('DAG_'+rmat_file,'a+')
# fo.write(str(num_v)+' '+ str(num_e)+'\n')
# fo.close()

with open('parsed_'+rmat_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(str(num_v)+' '+ str(num_e) + '\n' + content)

