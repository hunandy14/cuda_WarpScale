# /*****************************************************************
# Name : 
# Date : 2018/05/29
# By   : CharlotteHonG
# Final: 2018/05/29
# *****************************************************************/
NVCCFLAGS :=
NVCCFLAGS += -O3
NVCCFLAGS += -std=c++11
NVCCFLAGS += -Xcompiler -fopenmp
OPENCV := `pkg-config opencv --libs --cflags`
CXXINC := -IOpenBMP

# ================================================================
all: bilinear_main.out
rebuild: clear all

run: bilinear_main.out bilinear_main.o bilinear.o OpenBMP.o
	./bilinear_main.out
rerun: clear run

clear:
	rm -f *.o *.out

# ================================================================
bilinear_main.out: bilinear_main.o bilinear.o OpenBMP.o
	nvcc $(NVCCFLAGS) *.o -o bilinear_main.out $(OPENCV) $(CXXINC)

bilinear_main.o: bilinear_main.cpp
	nvcc $(NVCCFLAGS) -c bilinear_main.cpp $(OPENCV) $(CXXINC)
bilinear.o: bilinear.cu bilinear.hpp
	nvcc $(NVCCFLAGS) -c bilinear.cu $(OPENCV) $(CXXINC)
OpenBMP.o: OpenBMP/OpenBMP.cpp OpenBMP/OpenBMP.hpp
	nvcc $(NVCCFLAGS) -c OpenBMP/OpenBMP.cpp $(OPENCV) $(CXXINC)
