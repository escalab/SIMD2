CUBOOL = cuBool
CUBOOL_INCLUDE = $(CUBOOL)/cubool/include/
CUBOOL_LIB = $(CUBOOL)/build/cubool/

default: __tc_baseline

__tc_baseline: cubool_tc.cpp
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:cuBool/build/cubool"
	g++ $< -o $@ -I$(CUBOOL_INCLUDE) -L$(CUBOOL_LIB) -lcubool

clean:
	rm __tc_baseline