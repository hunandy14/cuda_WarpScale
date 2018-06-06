/*****************************************************************
Name : 
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
*****************************************************************/
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "OpenBMP.hpp"
#include "CudaMem\CudaMem.cuh"
#include "Timer.hpp"

__host__ void cuWarpScale_rgb(const ImgData & src, ImgData & dst, double ratio);
void WarpScale_rgb(const basic_ImgData & src, basic_ImgData & dst, double ratio);



