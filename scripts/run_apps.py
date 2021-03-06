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

def run_aplp_baseline(f):
    input_data = open(str(f))
    command = ['./../apps/aplp/ecl-aplp/ecl-aplp']
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
    print('Appliation Datasize baseline(ms) cuASR(ms) emulation(ms) iterations sparseOverdhead sparse(ms)')
    # aplp
    data_dir = '../apps/data/aplp_input/'
    aplp_baseline = run_aplp_baseline(data_dir+'DAG_rmat_4096.txt')
    aplp_cuAsr = run_aplp(data_dir+'DAG_rmat_4096.txt', 'cuASR')
    aplp_emulation = run_aplp(data_dir+'DAG_rmat_4096.txt', 'emulation')
    itr = get_itr(aplp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'DAG_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('APLP-Small:', 4096,aplp_baseline, aplp_cuAsr, aplp_emulation, sparse_total, sparse_kernel)

    aplp_baseline = run_aplp_baseline(data_dir+'DAG_rmat_8192.txt')
    aplp_cuAsr = run_aplp(data_dir+'DAG_rmat_8192.txt', 'cuASR')
    aplp_emulation = run_aplp(data_dir+'DAG_rmat_8192.txt', 'emulation')
    itr = get_itr(aplp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'DAG_rmat_8192.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('APLP-Medium:', 8192,aplp_baseline, aplp_cuAsr, aplp_emulation, sparse_total, sparse_kernel)

    aplp_baseline = run_aplp_baseline(data_dir+'DAG_rmat_16284.txt')
    aplp_cuAsr = run_aplp(data_dir+'DAG_rmat_16284.txt', 'cuASR')
    aplp_emulation = run_aplp(data_dir+'DAG_rmat_16284.txt', 'emulation')
    itr = get_itr(aplp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'DAG_rmat_16284.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('APLP-Large:', 16384,aplp_baseline, aplp_cuAsr, aplp_emulation, sparse_total, sparse_kernel)

    # mst

    data_dir = '../apps/data/mst_input/'
    mst_baseline = run_mst(data_dir+'mst_rmat_1024.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'mst_rmat_1024.txt','cuASR')
    mst_emulation = run_mst(data_dir+'mst_rmat_1024.txt','emulation')
    itr = get_itr(mst_emulation)
    emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_1024.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MST-Small:', 1024,mst_baseline, mst_cuAsr, mst_emulation, sparse_total, sparse_kernel)

    mst_baseline = run_mst(data_dir+'mst_rmat_2048.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'mst_rmat_2048.txt','cuASR')
    mst_emulation = run_mst(data_dir+'mst_rmat_2048.txt','emulation')
    itr = get_itr(mst_emulation)
    emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_2048.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MST-Medium:', 2048,mst_baseline, mst_cuAsr, mst_emulation, sparse_total, sparse_kernel)

    
    mst_baseline = run_mst(data_dir+'mst_rmat_4096.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'mst_rmat_4096.txt','cuASR')
    mst_emulation = run_mst(data_dir+'mst_rmat_4096.txt','emulation')
    itr = get_itr(mst_emulation)
    emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MST-Large:', 4096 ,mst_baseline, mst_cuAsr, mst_emulation, sparse_total, sparse_kernel)
    
    #maxrp

    data_dir = '../apps/data/maxrp_input/'
    maxrp_baseline = run_maxrp(data_dir+'maxrp_rmat_4096.txt', 'baseline')
    maxrp_cuAsr = run_maxrp(data_dir+'maxrp_rmat_4096.txt', 'cuASR')
    maxrp_emulation = run_maxrp(data_dir+'maxrp_rmat_4096.txt', 'emulation')
    itr = get_itr(maxrp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'maxrp_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MAXRP-Small:', 4096,maxrp_baseline, maxrp_cuAsr, maxrp_emulation, sparse_total, sparse_kernel)

    maxrp_baseline = run_maxrp(data_dir+'maxrp_rmat_8192.txt', 'baseline')
    maxrp_cuAsr = run_maxrp(data_dir+'maxrp_rmat_8192.txt', 'cuASR')
    maxrp_emulation = run_maxrp(data_dir+'maxrp_rmat_8192.txt', 'emulation')
    itr = get_itr(maxrp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'maxrp_rmat_8192.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MAXRP-Medium:', 8192,maxrp_baseline, maxrp_cuAsr, maxrp_emulation, sparse_total, sparse_kernel)

    maxrp_baseline = run_maxrp(data_dir+'maxrp_rmat_16384.txt', 'baseline')
    maxrp_cuAsr = run_maxrp(data_dir+'maxrp_rmat_16384.txt', 'cuASR')
    maxrp_emulation = run_maxrp(data_dir+'maxrp_rmat_16384.txt', 'emulation')
    itr = get_itr(maxrp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'maxrp_rmat_16384.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MAXRP-Large:', 16384,maxrp_baseline, maxrp_cuAsr, maxrp_emulation, sparse_total, sparse_kernel)

    #minrp

    data_dir = '../apps/data/minrp_input/'
    minrp_baseline = run_minrp(data_dir+'minrp_rmat_4096.txt', 'baseline')
    minrp_cuAsr = run_minrp(data_dir+'minrp_rmat_4096.txt', 'cuASR')
    minrp_emulation = run_minrp(data_dir+'minrp_rmat_4096.txt', 'emulation')
    itr = get_itr(minrp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'minrp_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MINRP-Small:', 4096,minrp_baseline, minrp_cuAsr, minrp_emulation, sparse_total, sparse_kernel)

    minrp_baseline = run_minrp(data_dir+'minrp_rmat_8192.txt', 'baseline')
    minrp_cuAsr = run_minrp(data_dir+'minrp_rmat_8192.txt', 'cuASR')
    minrp_emulation = run_minrp(data_dir+'minrp_rmat_8192.txt', 'emulation')
    itr = get_itr(minrp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'minrp_rmat_8192.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MINRP-Medium:', 8192,minrp_baseline, minrp_cuAsr, minrp_emulation, sparse_total, sparse_kernel)

    minrp_baseline = run_minrp(data_dir+'minrp_rmat_16384.txt', 'baseline')
    minrp_cuAsr = run_minrp(data_dir+'minrp_rmat_16384.txt', 'cuASR')
    minrp_emulation = run_minrp(data_dir+'minrp_rmat_16384.txt', 'emulation')
    itr = get_itr(minrp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'minrp_rmat_16384.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MINRP-Large:', 16384,minrp_baseline, minrp_cuAsr, minrp_emulation, sparse_total, sparse_kernel)
    
    # mcp
    data_dir = '../apps/data/rmat_data/'
    mcp_baseline = run_mcp(data_dir+'parsed_rmat_4096.txt', 'baseline')
    mcp_cuAsr = run_mcp(data_dir+'parsed_rmat_4096.txt', 'cuASR')
    mcp_emulation = run_mcp(data_dir+'parsed_rmat_4096.txt', 'emulation')
    itr = get_itr(mcp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MCP-Small:', 4096, mcp_baseline, mcp_cuAsr, mcp_emulation, sparse_total, sparse_kernel)

    mcp_baseline = run_mcp(data_dir+'parsed_rmat_8192.txt', 'baseline')
    mcp_cuAsr = run_mcp(data_dir+'parsed_rmat_8192.txt', 'cuASR')
    mcp_emulation = run_mcp(data_dir+'parsed_rmat_8192.txt', 'emulation')
    itr = get_itr(mcp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_8192.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MCP-Medium:', 8192, mcp_baseline, mcp_cuAsr, mcp_emulation, sparse_total, sparse_kernel)

    mcp_baseline = run_mcp(data_dir+'parsed_rmat_16384.txt', 'baseline')
    mcp_cuAsr = run_mcp(data_dir+'parsed_rmat_16384.txt', 'cuASR')
    mcp_emulation = run_mcp(data_dir+'parsed_rmat_16384.txt', 'emulation')
    itr = get_itr(mcp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_16384.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('MCP-Large:', 16384, mcp_baseline, mcp_cuAsr, mcp_emulation, sparse_total, sparse_kernel)

    # gtc
    gtc_baseline = run_gtc(1024,0.0001, 'baseline')
    gtc_cuAsr = run_gtc(1024,0.0001, 'cuASR')
    gtc_emulation = run_gtc(1024,0.0001, 'emulation')
    itr = get_itr(gtc_emulation)
    emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_1024.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('GTC-SMALL:', 1024,gtc_baseline, gtc_cuAsr, gtc_emulation, sparse_total, sparse_kernel)

    gtc_baseline = run_gtc(4096,0.0001, 'baseline')
    gtc_cuAsr = run_gtc(4096,0.0001, 'cuASR')
    gtc_emulation = run_gtc(4096,0.0001, 'emulation')
    itr = get_itr(gtc_emulation)
    emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('GTC-Medium:', 4096 ,gtc_baseline, gtc_cuAsr, gtc_emulation, sparse_total, sparse_kernel)

    gtc_baseline = run_gtc(8192,0.0001, 'baseline')
    gtc_cuAsr = run_gtc(8192,0.0001, 'cuASR')
    gtc_emulation = run_gtc(8192,0.0001, 'emulation')
    itr = get_itr(gtc_emulation)
    emulation_sparse = run_cusparselt(itr, '../apps/data/rmat_data/parsed_rmat_8192.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('GTC-Large:', 8192 ,gtc_baseline, gtc_cuAsr, gtc_emulation, sparse_total, sparse_kernel)


    # apsp
    data_dir = "../apps/data/rmat_data/"
    apsp_baseline = run_apsp_baseline(data_dir+'parsed_rmat_4096.txt')
    apsp_cuAsr = run_apsp(data_dir+'parsed_rmat_4096.txt', 'cuASR')
    apsp_emulation = run_apsp(data_dir+'parsed_rmat_4096.txt', 'emulation')
    itr = get_itr(apsp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('APSP-Small:', 4096,apsp_baseline, apsp_cuAsr, apsp_emulation, sparse_total, sparse_kernel)

    apsp_baseline = run_apsp_baseline(data_dir+'parsed_rmat_8192.txt')
    apsp_cuAsr = run_apsp(data_dir+'parsed_rmat_8192.txt', 'cuASR')
    apsp_emulation = run_apsp(data_dir+'parsed_rmat_8192.txt', 'emulation')
    itr = get_itr(apsp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_8192.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('APSP-Medium:', 8192,apsp_baseline, apsp_cuAsr, apsp_emulation, sparse_total, sparse_kernel)

    apsp_baseline = run_apsp_baseline(data_dir+'parsed_rmat_16384.txt')
    apsp_cuAsr = run_apsp(data_dir+'parsed_rmat_16384.txt', 'cuASR')
    apsp_emulation = run_apsp(data_dir+'parsed_rmat_16384.txt', 'emulation')
    itr = get_itr(apsp_emulation)
    emulation_sparse = run_cusparselt(itr, data_dir+'parsed_rmat_16384.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('APSP-Large:', 16384,apsp_baseline, apsp_cuAsr, apsp_emulation, sparse_total, sparse_kernel)

 

   
    # pld
    data_dir = '../apps/data/rmat_data/'
    pld_baseline = run_pld_gen(4096,4096,4096,10,'baseline')
    pld_cuAsr = run_pld_gen(4096,4096,4096,10,'cuASR')
    pld_emulation = run_pld_gen(4096,4096,4096,10,'emulation')
    emulation_sparse = run_cusparselt(1, data_dir+'parsed_rmat_4096.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('KNN-SMALL:', 4096**2,pld_baseline, pld_cuAsr, pld_emulation, sparse_total, sparse_kernel)

    pld_baseline = run_pld_gen(8192,8192,8192,10,'baseline')
    pld_cuAsr = run_pld_gen(8192,8192,8192,10,'cuASR')
    pld_emulation = run_pld_gen(8192,8192,8192,10,'emulation')
    emulation_sparse = run_cusparselt(1, data_dir+'parsed_rmat_8192.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('KNN-Medium:', 8192**2,pld_baseline, pld_cuAsr, pld_emulation, sparse_total, sparse_kernel)

    
    pld_baseline = run_pld_gen(16384,16384,16384,10,'baseline')
    pld_cuAsr = run_pld_gen(16384,16384,16384,10,'cuASR')
    pld_emulation = run_pld_gen(16384,16384,16384,10,'emulation')
    emulation_sparse = run_cusparselt(1, data_dir+'parsed_rmat_16384.txt')
    emulation_sparse = emulation_sparse.split(' ')
    sparse_total = emulation_sparse[0]
    sparse_kernel = emulation_sparse[1]
    print('KNN-Large:', 16384**2, pld_baseline, pld_cuAsr, pld_emulation, sparse_total, sparse_kernel)



if __name__ == "__main__":
	main()