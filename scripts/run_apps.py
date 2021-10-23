import sys
import os
import subprocess

data_size = [1024, 8192, 32768]

def run_apsp_gen(v,d,s,version):
    command = ['./../apps/apsp/'+version + '/apsp_'+version, str(v), str(d), str(s)]
    res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_pld_gen(rn,qn,d,k,version):
    command = ['./../apps/pld/'+version + '/knn_'+version, str(rn), str(qn), str(d), str(k)]
    res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    return res.split('\n')[0]

def run_mst(f,version):
    
    if version == 'baseline':
        command = ['./../apps/mst/'+version + '/mst_'+version, '-r' '10',str(f)]
        res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        res = res.split('\n')
        res = res[1:]
        total_time = 0
        # print(res)
        for line in res:
            # print(line)
            if line.startswith('prepare') or not line: continue
            else: total_time += float(line[line.rindex('PBBS-time:')+10:]) 
        return float("{:.5f}".format(total_time * 1000))

    else:
        command = ['./../apps/mst/'+version + '/mst_'+version, '-f',str(f)]
        res = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
        return res.split('\n')[0]

def main():

    # apsp
    apsp_baseline = run_apsp_gen(512,0.01,10, 'baseline')
    apsp_cuAsr = run_apsp_gen(512,0.01,10, 'cuASR')
    apsp_emulation = run_apsp_gen(512,0.01,10, 'emulation')
    print('APSP-SMALL:', 512**2,apsp_baseline, apsp_cuAsr, apsp_emulation)

    apsp_baseline = run_apsp_gen(4096,0.01,10, 'baseline')
    apsp_cuAsr = run_apsp_gen(4096,0.01,10, 'cuASR')
    apsp_emulation = run_apsp_gen(4096,0.01,10, 'emulation')
    print('APSP-Medium:', 4096**2,apsp_baseline, apsp_cuAsr, apsp_emulation)

    apsp_baseline = run_apsp_gen(8192,0.01,10, 'baseline')
    apsp_cuAsr = run_apsp_gen(8192,0.01,10, 'cuASR')
    apsp_emulation = run_apsp_gen(8192,0.01,10, 'emulation')
    print('APSP-Medium-Large:', 8192**2,apsp_baseline, apsp_cuAsr, apsp_emulation)

    apsp_baseline = run_apsp_gen(16384,0.01,10, 'baseline')
    apsp_cuAsr = run_apsp_gen(16384,0.01,10, 'cuASR')
    apsp_emulation = run_apsp_gen(16384,0.01,10, 'emulation')
    print('APSP-Large:', 16384**2, apsp_baseline, apsp_cuAsr, apsp_emulation)

    # pld
    pld_baseline = run_pld_gen(4096,4096,4096,10,'baseline')
    pld_cuAsr = run_pld_gen(4096,4096,4096,10,'cuASR')
    pld_emulation = run_pld_gen(4096,4096,4096,10,'emulation')
    print('KNN-SMALL:', 4096**2,pld_baseline, pld_cuAsr, pld_emulation)

    pld_baseline = run_pld_gen(8192,8192,8192,10,'baseline')
    pld_cuAsr = run_pld_gen(8192,8192,8192,10,'cuASR')
    pld_emulation = run_pld_gen(8192,8192,8192,10,'emulation')
    print('KNN-Medium:', 8192**2,pld_baseline, pld_cuAsr, pld_emulation)

    pld_baseline = run_pld_gen(16384,16384,16384,10,'baseline')
    pld_cuAsr = run_pld_gen(16384,16384,16384,10,'cuASR')
    pld_emulation = run_pld_gen(16384,16384,16384,10,'emulation')
    print('KNN-Large:', 16384**2, pld_baseline, pld_cuAsr, pld_emulation)

    # # mst

    data_dir = '../apps/data/'
    mst_baseline = run_mst(data_dir+'1024_0.5_dense_pbbs.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'1024_0.5_dense_pbbs.txt','cuASR')
    mst_emulation = run_mst(data_dir+'1024_0.5_dense_pbbs.txt','emulation')
    print('MST-SMALL:', 1024**2,mst_baseline, mst_cuAsr, mst_emulation)

    mst_baseline = run_mst(data_dir+'2048_0.5_dense_pbbs.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'2048_0.5_dense_pbbs.txt','cuASR')
    mst_emulation = run_mst(data_dir+'2048_0.5_dense_pbbs.txt','emulation')
    print('MST-SMALL-Medium:', 2048**2,mst_baseline, mst_cuAsr, mst_emulation)

    mst_baseline = run_mst(data_dir+'3072_0.5_dense_pbbs.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'3072_0.5_dense_pbbs.txt','cuASR')
    mst_emulation = run_mst(data_dir+'3072_0.5_dense_pbbs.txt','emulation')
    print('MST-SMALL-Medium:', 3072**2,mst_baseline, mst_cuAsr, mst_emulation)

    mst_baseline = run_mst(data_dir+'4096_0.5_dense_pbbs.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'4096_0.5_dense_pbbs.txt','cuASR')
    mst_emulation = run_mst(data_dir+'4096_0.5_dense_pbbs.txt','emulation')
    print('MST-SMALL-Medium:', 4096**2,mst_baseline, mst_cuAsr, mst_emulation)

    mst_baseline = run_mst(data_dir+'8192_0.5_dense_pbbs.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'8192_0.5_dense_pbbs.txt','cuASR')
    mst_emulation = run_mst(data_dir+'8192_0.5_dense_pbbs.txt','emulation')
    print('MST-Medium:', 8192**2,mst_baseline, mst_cuAsr, mst_emulation)

    mst_baseline = run_mst(data_dir+'16384_0.5_dense_pbbs.txt','baseline')
    mst_cuAsr = run_mst(data_dir+'16384_0.5_dense_pbbs.txt','cuASR')
    mst_emulation = run_mst(data_dir+'16384_0.5_dense_pbbs.txt','emulation')
    print('MST-Large:', 16384**2, mst_baseline, mst_cuAsr, mst_emulation)

if __name__ == "__main__":
	main()