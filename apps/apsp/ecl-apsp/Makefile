all: ecl-apsp

ecl-apsp: ECL-APSP_10.cu ECLgraph.h
	nvcc -O3 -arch=sm_80 ECL-APSP_10.cu -o ecl-apsp

clean:
	rm -f ecl-apsp
