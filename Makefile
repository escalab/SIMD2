CUDADIR = /usr/local/cuda-11.7/
EMUDIRS = apps/apsp/emulation \
			apps/aplp/emulation \
			apps/maxrp/emulation \
			apps/minrp/emulation \
			apps/gtc/emulation \
			apps/mst/emulation \
			apps/pld/emulation \
			apps/mcp/emulation 
CUASRDIRS = apps/apsp/cuASR \
			apps/aplp/cuASR \
			apps/maxrp/cuASR \
			apps/minrp/cuASR \
			apps/gtc/cuASR \
			apps/mst/cuASR \
			apps/pld/cuASR \
			apps/mcp/cuASR 
BASEDIRS =	apps/apsp/ecl-apsp \
			apps/aplp/ecl-aplp \
			apps/maxrp/baseline \
			apps/minrp/baseline \
			apps/gtc/baseline \
			apps/mst/baseline \
			apps/pld/baseline \
			apps/mcp/baseline \

all: cuasr baseline emulation microbench sparse
cuasr: $(CUASRDIRS)
$(CUASRDIRS):
	$(MAKE) -C $@

baseline: $(BASEDIRS)
$(BASEDIRS):
	$(MAKE) -C $@

emulation: $(EMUDIRS)
$(EMUDIRS):
	$(MAKE) -C $@

microbench:
	make -C micro_bench

sparse:
	make -C apps/emulation_sparse/

test:
	make -C tests

.PHONY: cuasr $(CUASRDIRS) baseline $(BASEDIRS) emulation $(EMUDIRS)