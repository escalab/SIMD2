import sys
import os
import subprocess
from subprocess import Popen, PIPE, STDOUT

data_size = [1024, 8192, 32768]

def run_apsp_gen(v,d,s,version):
    command = ['./../apps/apsp/'+version + '/apsp_'+version, str(v), str(d), str(s)]
    res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_apsp(f,version):
    input_data = open(str(f))
    command = ['./../apps/apsp/'+version + '/apsp_'+version, '-f']
    res = subprocess.run(command, stdin = input_data, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_apsp_baseline(f):
    input_data = open(str(f))
    command = ['./../apps/apsp/ecl-apsp/ecl-apsp']
    res = subprocess.run(command, stdin = input_data, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_aplp(f,version):
    input_data = open(str(f))
    command = ['./../apps/aplp/'+version + '/aplp_'+version]
    res = subprocess.run(command, stdin = input_data, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_pld_gen(rn,qn,d,k,version):
    command = ['./../apps/pld/'+version + '/knn_'+version, str(rn), str(qn), str(d), str(k)]
    res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_mcp(f,version):
    input_data = open(str(f))
    command = ['./../apps/mcp/'+version + '/mcp_'+version]
    res = subprocess.run(command, stdin = input_data, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_maxrp(f,version):
    input_data = open(str(f))
    command = ['./../apps/maxrp/'+version + '/maxrp_'+version]
    res = subprocess.run(command, stdin = input_data, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_minrp(f,version):
    input_data = open(str(f))
    command = ['./../apps/minrp/'+version + '/minrp_'+version]
    res = subprocess.run(command, stdin = input_data, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_gtc(v,d,version):
    if version == 'baseline':
        res = os.popen("cd ../apps/gtc/baseline;"+"./gtc_baseline "+str(v)+' ' + str(d)).read()
        # os.system("./gtc_baseline "+str(v)+' ' + str(d))
        # initcmd = ['export', "LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:/nfshome/yuz057/SIMD2/apps/gtc/baseline/cuBool/build/cubool\""]
        # subprocess.run(initcmd, stdout=subprocess.PIPE).stdout.decode('utf-8')
        return res.split('\n')[0]
    else:
        command = ['./../apps/gtc/'+version + '/gtc_'+version, str(v), str(d)]
        res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_mst(f,version):
    
    if version == 'baseline':
        command = ['./../apps/mst/'+version + '/mst_'+version, '-r' '10',str(f)]
        res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        res = res.split('\n')
        res = res[1:]
        total_time = 0
        for line in res:
            if line.startswith('prepare') or not line: continue
            else: total_time += float(line[line.rindex('PBBS-time:')+10:]) 
        return float("{:.5f}".format(total_time * 1000))

    else:
        command = ['./../apps/mst/'+version + '/mst_'+version, '-f',str(f)]
        res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        return res.split('\n')[0]

def run_cusparselt(itr,f):
    input_data = open(str(f))
    command = ['./../apps/emulation_sparse/emulation_sparse', str(itr)]
    res = subprocess.run(command, stdin = input_data, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def get_itr(res):
    return int(res.split(' ')[-1])

def main():

    # aplp
    data_dir = '../apps/data/aplp_input/'
    # aplp_baseline = run_aplp(data_dir+'DAG_rmat_4096.txt', 'baseline')
    # aplp_cuAsr = run_aplp(data_dir+'DAG_rmat_4096.txt', 'cuASR')
    aplp_emulation = run_aplp(data_dir+'DAG_rmat_4096.txt', 'emulation')
    # itr = get_itr(aplp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'DAG_rmat_4096.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('APLP-Small:',aplp_emulation)

    # aplp_baseline = run_aplp(data_dir+'DAG_rmat_8192.txt', 'baseline')
    # aplp_cuAsr = run_aplp(data_dir+'DAG_rmat_8192.txt', 'cuASR')
    aplp_emulation = run_aplp(data_dir+'DAG_rmat_8192.txt', 'emulation')
    # itr = get_itr(aplp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'DAG_rmat_8192.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('APLP-Medium:', 8192,aplp_emulation)

    # aplp_baseline = run_aplp(data_dir+'DAG_rmat_16284.txt', 'baseline')
    # aplp_cuAsr = run_aplp(data_dir+'DAG_rmat_16284.txt', 'cuASR')
    aplp_emulation = run_aplp(data_dir+'DAG_rmat_16284.txt', 'emulation')
    # itr = get_itr(aplp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'DAG_rmat_16284.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('APLP-Large:', 16384,aplp_emulation)

    # mst

    data_dir = '../apps/data/mst_input/'
    # mst_baseline = run_mst(data_dir+'mst_rmat_1024.txt','baseline')
    # mst_cuAsr = run_mst(data_dir+'mst_rmat_1024.txt','cuASR')
    mst_emulation = run_mst(data_dir+'mst_rmat_1024.txt','emulation')
    # itr = get_itr(mst_emulation)
    # emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_1024.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MST-Small:', 1024, mst_emulation)

    # mst_baseline = run_mst(data_dir+'mst_rmat_2048.txt','baseline')
    # mst_cuAsr = run_mst(data_dir+'mst_rmat_2048.txt','cuASR')
    mst_emulation = run_mst(data_dir+'mst_rmat_2048.txt','emulation')
    # itr = get_itr(mst_emulation)
    # emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_2048.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MST-Medium:', 2048,mst_emulation)

    
    # mst_baseline = run_mst(data_dir+'mst_rmat_4096.txt','baseline')
    # mst_cuAsr = run_mst(data_dir+'mst_rmat_4096.txt','cuASR')
    mst_emulation = run_mst(data_dir+'mst_rmat_4096.txt','emulation')
    # itr = get_itr(mst_emulation)
    # emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_4096.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MST-Large:', 4096 ,mst_emulation)
    
    #maxrp

    data_dir = '../apps/data/maxrp_input/'
    # maxrp_baseline = run_maxrp(data_dir+'maxrp_rmat_4096.txt', 'baseline')
    # maxrp_cuAsr = run_maxrp(data_dir+'maxrp_rmat_4096.txt', 'cuASR')
    maxrp_emulation = run_maxrp(data_dir+'maxrp_rmat_4096.txt', 'emulation')
    # itr = get_itr(maxrp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'maxrp_rmat_4096.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MAXRP-Small:', 4096,maxrp_emulation)

    # maxrp_baseline = run_maxrp(data_dir+'maxrp_rmat_8192.txt', 'baseline')
    # maxrp_cuAsr = run_maxrp(data_dir+'maxrp_rmat_8192.txt', 'cuASR')
    maxrp_emulation = run_maxrp(data_dir+'maxrp_rmat_8192.txt', 'emulation')
    # itr = get_itr(maxrp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'maxrp_rmat_8192.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MAXRP-Medium:', 8192, maxrp_emulation)

    # maxrp_baseline = run_maxrp(data_dir+'maxrp_rmat_16384.txt', 'baseline')
    # maxrp_cuAsr = run_maxrp(data_dir+'maxrp_rmat_16384.txt', 'cuASR')
    maxrp_emulation = run_maxrp(data_dir+'maxrp_rmat_16384.txt', 'emulation')
    # itr = get_itr(maxrp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'maxrp_rmat_16384.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MAXRP-Large:', 16384,maxrp_emulation)

    #minrp

    data_dir = '../apps/data/minrp_input/'
    # minrp_baseline = run_minrp(data_dir+'minrp_rmat_4096.txt', 'baseline')
    # minrp_cuAsr = run_minrp(data_dir+'minrp_rmat_4096.txt', 'cuASR')
    minrp_emulation = run_minrp(data_dir+'minrp_rmat_4096.txt', 'emulation')
    # itr = get_itr(minrp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'minrp_rmat_4096.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MINRP-Small:', 4096,minrp_emulation)

    # minrp_baseline = run_minrp(data_dir+'minrp_rmat_8192.txt', 'baseline')
    # minrp_cuAsr = run_minrp(data_dir+'minrp_rmat_8192.txt', 'cuASR')
    minrp_emulation = run_minrp(data_dir+'minrp_rmat_8192.txt', 'emulation')
    # itr = get_itr(minrp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'minrp_rmat_8192.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MINRP-Medium:', 8192,minrp_emulation)

    # minrp_baseline = run_minrp(data_dir+'minrp_rmat_16384.txt', 'baseline')
    # minrp_cuAsr = run_minrp(data_dir+'minrp_rmat_16384.txt', 'cuASR')
    minrp_emulation = run_minrp(data_dir+'minrp_rmat_16384.txt', 'emulation')
    # itr = get_itr(minrp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'minrp_rmat_16384.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MINRP-Large:', 16384,minrp_emulation)
    
    # mcp
    data_dir = '../apps/data/rmat_data/'
    # mcp_baseline = run_mcp(data_dir+'parsed_rmat_4096.txt', 'baseline')
    # mcp_cuAsr = run_mcp(data_dir+'parsed_rmat_4096.txt', 'cuASR')
    mcp_emulation = run_mcp(data_dir+'parsed_rmat_4096.txt', 'emulation')
    # itr = get_itr(mcp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_4096.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MCP-Small:', 4096, mcp_emulation)

    # mcp_baseline = run_mcp(data_dir+'parsed_rmat_8192.txt', 'baseline')
    # mcp_cuAsr = run_mcp(data_dir+'parsed_rmat_8192.txt', 'cuASR')
    mcp_emulation = run_mcp(data_dir+'parsed_rmat_8192.txt', 'emulation')
    # itr = get_itr(mcp_emulation)
    # emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_8192.txt')
    # emulation_sparse = emulation_sparse.split(' ')
    # sparse_total = emulation_sparse[0]
    # sparse_kernel = emulation_sparse[1]
    print('MCP-Medium:', 8192, mcp_emulation)

    # mcp_baseline = run_mcp(data_dir+'parsed_rmat_16384.txt', 'baseline')
    # mcp_cuAsr = run_mcp(data_dir+'parsed_rmat_16384.txt', 'cuASR')
    mcp_emulation = run_mcp(data_dir+'parsed_rmat_16384.txt', 'emulation')
    print('MCP-Large:', 16384,mcp_emulation)

    # gtc
    gtc_emulation = run_gtc(1024,0.0001, 'emulation')
    print('GTC-SMALL:', 1024, gtc_emulation)

    gtc_emulation = run_gtc(4096,0.0001, 'emulation')
    print('GTC-Medium:', 4096 , gtc_emulation)

    gtc_emulation = run_gtc(8192,0.0001, 'emulation')
    print('GTC-Large:', 8192 , gtc_emulation)


    # apsp
    data_dir = "../apps/data/rmat_data/"
    apsp_emulation = run_apsp(data_dir+'parsed_rmat_4096.txt', 'emulation')

    print('APSP-Small:', 4096, apsp_emulation)

    apsp_emulation = run_apsp(data_dir+'parsed_rmat_8192.txt', 'emulation')
    print('APSP-Medium:', 8192, apsp_emulation)

    apsp_emulation = run_apsp(data_dir+'parsed_rmat_16384.txt', 'emulation')
    print('APSP-Large:', 16384, apsp_emulation)



if __name__ == "__main__":
	main()