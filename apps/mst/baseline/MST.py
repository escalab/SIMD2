'''
starter script for mst baseline
'''
import subprocess
import sys

def main():
    input_data = sys.argv[1]
    try:
        rounds = sys.argv[2]
    except: rounds = str(10)
    # print(input_data,rounds)
    command = ['./mst_baseline','-r', rounds, input_data]
    res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    res = res.split('\n')
    res = res[1:]
    total_time = 0
    for line in res:
        if line.startswith('prepare') or not line: continue
        else: total_time += float(line[line.rindex('PBBS-time:')+10:]) 
    print("{:.5f}".format(total_time/int(rounds) * 1000))
if __name__ == "__main__":
	main()