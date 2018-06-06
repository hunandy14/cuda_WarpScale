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

using std::vector;

__host__ double biliner_share(vector<float>& dst, const vector<float>& src,
	size_t width, size_t height, float ratio);

__host__ double biliner_CPU(vector<float>& dst, const vector<float>& src,
	size_t width, size_t height, float ratio);

__global__ void cuWarpScale_rgb(const cubasic_ImgData & src, cubasic_ImgData & dst, double ratio);

__host__ void WarpScale_rgb(const basic_ImgData & src, basic_ImgData & dst, double ratio);