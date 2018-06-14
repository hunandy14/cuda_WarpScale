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
	cuImgData() = default;
	cuImgData(const basic_ImgData& src) :
		CudaData(src.raw_img.data(), src.raw_img.size()),
		width(src.width), height(src.height), bits(src.bits) {}
	cuImgData(uint32_t w, uint32_t h, uint16_t bits): 
		CudaData(w*h * bits>>3), width(w), height(h), bits(bits){}
	~cuImgData() = default;
public:
	void out(basic_ImgData& dst) {
		dst.raw_img.resize(width*height * bits>>3);
		dst.width  = width;
		dst.height = height;
		dst.bits   = bits;
		memcpyOut(dst.raw_img.data(), dst.raw_img.size());
	}
	void resize(uint32_t w, uint32_t h, uint16_t bits) {
		this->~cuImgData();
		malloc(w*h * bits>>3);

		this->width  = w;
		this->height = h;
		this->bits   = bits;
	}
public:
	uint32_t width;
	uint32_t height;
	uint16_t bits;
};

__host__ void cuWarpScale_kernel_test(const basic_ImgData & src, basic_ImgData & dst, double ratio);
__host__ void imgSub(cuImgData & uSrc, const cuImgData & uDst);
__host__ void WarpScale_rgb(const cuImgData & uSrc, cuImgData & uDst, double ratio);
__host__ void WarpScale_rgb(const basic_ImgData & src, basic_ImgData & dst, double ratio);



