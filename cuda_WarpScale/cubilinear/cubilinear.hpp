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
	~cuImgData() = default;

	explicit cuImgData(const basic_ImgData& src) :
		CudaData(src.raw_img.data(), src.raw_img.size()),
		width(src.width), height(src.height), bits(src.bits) {}
	cuImgData(uint32_t w, uint32_t h, uint16_t bits): 
		CudaData(w*h * bits>>3), width(w), height(h), bits(bits){}
public:
	void in(const basic_ImgData& dst) {
		int size = dst.raw_img.size();
		malloc(size);
		memcpyIn(dst.raw_img.data(), size);

		width  = dst.width;
		height = dst.height;
		bits   = dst.bits;
	}
	void out(basic_ImgData& dst) const {
		dst.raw_img.resize(width*height * bits>>3);
		dst.width  = width;
		dst.height = height;
		dst.bits   = bits;
		memcpyOut(dst.raw_img.data(), dst.raw_img.size());
	}
	void resize(uint32_t w, uint32_t h, uint16_t bits) {
		// 空間不足重new
		if(w*h > this->len) {
			this->~cuImgData();
			malloc(w*h * bits>>3);
			//cout << "reNewSize" << endl;
		} 
		// 空間充足直接用
		else if(w*h <= this->len) {
			//cout << "non reNewSize" << endl;
		}
		this->width  = w;
		this->height = h;
		this->bits   = bits;
	}
	void resize(const cuImgData& src) {
		resize(src.width, src.height, src.bits);
	}
public:
	int size(){
		return width*height*bits>>3;
	}
	int sizePix(){
		return width*height;
	}
public:
	uint32_t width;
	uint32_t height;
	uint16_t bits;
};

// 複製圖片
__host__ void imgCopy(const cuImgData & uSrc, cuImgData & uDst);

// 線性插補
__host__ void WarpScale_rgb(const cuImgData & uSrc, cuImgData & uDst, double ratio);
__host__ void WarpScale_rgb(const basic_ImgData & src, basic_ImgData & dst, double ratio);
__host__ void cuWarpScale_kernel_test(const basic_ImgData & src, basic_ImgData & dst, double ratio);

// 圖片相減
__host__ void imgSub(cuImgData & uSrc, const cuImgData & uDst);
// 圖片相加
__host__ void imgAdd(cuImgData & uSrc, const cuImgData & uDst);
// 高斯模糊
__host__ void GaussianBlur(const cuImgData & uSrc, cuImgData & uDst, int matLen, double sigma=0);

// 金字塔混和
__host__ void imgBlendHalf(const cuImgData & uimgA, const cuImgData & uimgB, cuImgData & uDst);
__host__ void imgBlendAlpha(const cuImgData & uimgA, const cuImgData & uimgB, cuImgData & uDst);
__host__ void mergeOverlap(const cuImgData & usrc1, const cuImgData & usrc2, const cuImgData & ublend, cuImgData & udst, vector<int> corner);
__host__ void getOverlap(const cuImgData & uSrc, const cuImgData & uSrc2, cuImgData & ucut1, cuImgData & ucut2, vector<int> corner);
// 圓柱投影
__host__ void WarpCylindrical(const cuImgData & uSrc, cuImgData & uDst, double R, int mx=0, int my=0, double edge=0.0);
__host__ void WarpCyliCorner(const cuImgData & uSrc, CudaData<int>& ucorner, int mx, int my);



