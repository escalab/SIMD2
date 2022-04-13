all: ecl-aplp

ecl-aplp: ECL-APSP_10.cu ECLgraph.h
	nvcc -O3 -arch=sm_80 ECL-APSP_10.cu -o ecl-aplp

clean:
	rm -f ecl-aplp
