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
#include "CudaMem.cuh"
#include "Timer.hpp"

// struct basic_ImgData {
// 	std::vector<unsigned char> raw_img;
// 	uint32_t width;
// 	uint32_t height;
// 	uint16_t bits;
// };
using uch = unsigned char;

class cuImgData:public CudaData<uch>{
public:
	cuImgData(const basic_ImgData& src) :
		CudaData(src.raw_img.data(), src.raw_img.size()),
		width(src.width), height(src.height) {}
	cuImgData(int size): CudaData(size) {}
public:
	void out(basic_ImgData& dst) {
		memcpyOut(dst.raw_img.data(), dst.raw_img.size());
	}
public:
	uint32_t width;
	uint32_t height;
	uint16_t bits;
};

__host__ void cuWarpScale_kernel_test(const basic_ImgData & src, basic_ImgData & dst, double ratio);
__host__ void WarpScale_rgb_test(const basic_ImgData & src, basic_ImgData & dst, double ratio);
//__host__ void WarpScale_rgb_test(const cuImgData & src, cuImgData & dst, double ratio);
__host__ void WarpScale_rgb(const basic_ImgData & src, basic_ImgData & dst, double ratio);



