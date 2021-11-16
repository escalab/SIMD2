CUBOOL = cuBool
CUBOOL_INCLUDE = $(CUBOOL)/cubool/include/
CUBOOL_LIB = $(CUBOOL)/build/cubool/

default: gtc_baseline

gtc_baseline: cubool_tc.cpp
	# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:cuBool/build/cubool"
	g++ $< -o $@ -I$(CUBOOL_INCLUDE) -L$(CUBOOL_LIB) -L. -lcubool

clean:
	rm gtc_baseline