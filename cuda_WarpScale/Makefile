# /*****************************************************************
# Name : 
# Date : 2018/05/29
# By   : CharlotteHonG
# Final: 2018/05/29
# *****************************************************************/
NVCCFLAGS :=
NVCCFLAGS += -std=c++11
NVCCFLAGS += -Xcompiler -fopenmp
OPENCV := `pkg-config opencv --libs --cflags`
CXXINC := -IOpenBMP -ICudaMem

# ================================================================
all: bilinear_main.out

run: bilinear_main.out
	./uda_copyArr.out
clear:
	rm -f *.o *.out

# ================================================================
bilinear_main.out: bilinear_main.o
	nvcc $(NVCCFLAGS) *.o -o bilinear_main.out $(OPENCV) $(CXXINC)


bilinear_main.o: bilinear_main.cu
	nvcc $(NVCCFLAGS) -c bilinear_main.cu $(OPENCV) $(CXXINC)
bilinear.o: bilinear.cu bilinear.cuh
	nvcc $(NVCCFLAGS) -c bilinear.cu $(OPENCV) $(CXXINC)
OpenBMP.o: OpenBMP/OpenBMP.cpp OpenBMP/OpenBMP.hpp
	nvcc $(NVCCFLAGS) -c OpenBMP/OpenBMP.cpp $(OPENCV) $(CXXINC)
