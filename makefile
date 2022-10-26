all: c15 MemTop

CXXARGS = -std=c++14 -g -ICUDASAMPLES/Common #-gencode arch=compute_35,code=sm_35
CXXARGSDEBUG = -Xcompiler -DNDEBUG
UNAME = $(shell uname)

ifeq ($(UNAME), Linux)
LIBS = -lcuda -lstdc++fs #-lnvidia-ml
else
LIBS = -lcuda
endif

debug: c15Debug MemTopDebug

c15: starter_other/ConstMemory.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) starter_other/ConstMemory.cu -o c15

c15Debug: starter_other/ConstMemory.cu
	nvcc $(CXXARGS) $(CXXARGSDEBUG) $(INCLUDES) $(LIBS) -DIsDebug starter_other/ConstMemory.cu -o c15

MemTop: capture.cu
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) capture.cu -o MemTop

MemTopDebug: capture.cu
	nvcc $(CXXARGS) $(CXXARGSDEBUG) $(INCLUDES) $(LIBS) -DIsDebug capture.cu -o MemTop

clean:
	rm -f *.o MemTop c15
