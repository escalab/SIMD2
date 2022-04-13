import os
import subprocess
from subprocess import Popen, PIPE, STDOUT


ncu_metrics = 'sm__pipe_tensor_cycles_active.avg,sm__inst_executed_pipe_tensor.avg,sm__inst_executed.avg,sm__cycles_active.avg'

base_command = ['ncu', '--metrics', ncu_metrics, '--section', 'SpeedOfLight', '--csv', '--log-file']

def run_apsp(o,f,version):
    input_data = open(str(f))
    spcommand = base_command + [o] +['./../apps/apsp/'+version + '/apsp_'+version, '-f']
    subprocess.run(spcommand, stdin = input_data)

def run_apsp_baseline(o,f):
    input_data = open(str(f))
    spcommand = base_command + [o] + ['./../apps/apsp/ecl-apsp/ecl-apsp']
    subprocess.run(spcommand, stdin = input_data)

def run_aplp(o, f,version):
    input_data = open(str(f))
    spcommand = base_command + [o] + ['./../apps/aplp/'+version + '/aplp_'+version]
    subprocess.run(spcommand, stdin = input_data)

def run_pld_gen(o, rn,qn,d,k,version):
    spcommand = base_command + [o] + ['./../apps/pld/'+version + '/knn_'+version, str(rn), str(qn), str(d), str(k)]
    subprocess.run(spcommand)

def run_mcp(o, f,version):
    input_data = open(str(f))
    spcommand = base_command + [o] + ['./../apps/mcp/'+version + '/mcp_'+version]
    subprocess.run(spcommand, stdin = input_data)

def run_maxrp(o, f,version):
    input_data = open(str(f))
    spcommand = base_command + [o] + ['./../apps/maxrp/'+version + '/maxrp_'+version]
    subprocess.run(spcommand, stdin = input_data)

def run_minrp(o, f,version):
    input_data = open(str(f))
    spcommand = base_command + [o] + ['./../apps/minrp/'+version + '/minrp_'+version]
    subprocess.run(spcommand, stdin = input_data)

def run_gtc(o,v,d,version):
    spcommand = base_command + [o] + ['./../apps/gtc/'+version + '/gtc_'+version, str(v), str(d)]
    subprocess.run(spcommand)

def run_mst(o, f,version):
    spcommand = base_command + [o] + ['./../apps/mst/'+version + '/mst_'+version, '-f',str(f)]
    subprocess.run(spcommand)


data_dir = "../apps/data/rmat_data/"
run_apsp('profile_result_apsp.csv',data_dir+'parsed_rmat_16384.txt', 'emulation')
run_apsp_baseline('profile_result_apsp_baseline.csv', data_dir+'parsed_rmat_16384.txt')

data_dir = '../apps/data/aplp_input/'
run_aplp('profile_result_aplp.csv', data_dir+'DAG_rmat_16284.txt', 'emulation')
run_aplp('profile_result_aplp_baseline.csv', data_dir+'DAG_rmat_16284.txt', 'baseline')

data_dir = '../apps/data/mst_input/'
run_mst('profile_result_mst.csv',data_dir+'mst_rmat_4096.txt','emulation')
## use prof in mst dir to profile mst baseline

data_dir = '../apps/data/maxrp_input/'
run_maxrp('profile_result_maxrp.csv', data_dir+'maxrp_rmat_16384.txt', 'emulation')
run_maxrp('profile_result_maxrp_baseline.csv', data_dir+'maxrp_rmat_16384.txt', 'baseline')

data_dir = '../apps/data/minrp_input/'
run_minrp('profile_result_minrp.csv',data_dir+'minrp_rmat_16384.txt', 'emulation')
run_minrp('profile_result_minrp_baseline.csv',data_dir+'minrp_rmat_16384.txt', 'baseline')

data_dir = '../apps/data/rmat_data/'
run_mcp('profile_result_mcp.csv',data_dir+'parsed_rmat_16384.txt', 'emulation')
run_mcp('profile_result_mcp_baseline.csv',data_dir+'parsed_rmat_16384.txt', 'baseline')

run_gtc('profile_result_gtc.csv',8192,0.0001, 'emulation')
## use prof in mst dir to profile gtc baseline

run_pld_gen('profile_result_pld.csv', 16384,16384,16384,10,'emulation')
run_pld_gen('profile_result_pld_baseline.csv', 16384,16384,16384,10,'baseline')