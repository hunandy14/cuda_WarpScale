# /*****************************************************************
# Name : 
# Date : 2018/05/29
# By   : CharlotteHonG
# Final: 2018/05/29
# *****************************************************************/
NVCCFLAGS :=
NVCCFLAGS += -O3
NVCCFLAGS += -std=c++14
NVCCFLAGS += -Xcompiler -fopenmp

OPENCV := `pkg-config opencv --libs --cflags`

CXXINC := -Iheader
CXXINC += -ILapBlend

# ================================================================
all: bilinear_main.out
rebuild: clear all

run: bilinear_main.out bilinear_main.o cubilinear.o
	./bilinear_main.out
rerun: clear run

clear:
	rm -f *.o *.out

# ================================================================
bilinear_main.out: bilinear_main.o cubilinear.o LapBlend.o
	nvcc $(NVCCFLAGS) *.o -o bilinear_main.out $(OPENCV) $(CXXINC)

bilinear_main.o: bilinear_main.cpp
	nvcc $(NVCCFLAGS) -c bilinear_main.cpp $(OPENCV) $(CXXINC)
cubilinear.o: LapBlend/cubilinear.cu LapBlend/cubilinear.hpp
	nvcc $(NVCCFLAGS) -c LapBlend/cubilinear.cu $(OPENCV) $(CXXINC)
LapBlend.o: LapBlend/LapBlend.cpp LapBlend/LapBlend.hpp
	nvcc $(NVCCFLAGS) -c LapBlend/LapBlend.cpp $(OPENCV) $(CXXINC)