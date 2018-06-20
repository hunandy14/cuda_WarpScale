/*****************************************************************
Name : 
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
*****************************************************************/
#pragma once
#include <utility>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "OpenBMP.hpp"
#include "Timer.hpp"

#include "CudaMem.cuh"

class cuImgData:public CudaData<unsigned char>{
public:
	cuImgData() = default;
	~cuImgData() = default;
	explicit cuImgData(const basic_ImgData& src) :
		CudaData(src.raw_img.data(), src.raw_img.size()),
		width(src.width), height(src.height), bits(src.bits) {}
	cuImgData(uint32_t w, uint32_t h, uint16_t bits): 
		CudaData(w*h * bits>>3), width(w), height(h), bits(bits) {}
public:
	cuImgData(const cuImgData& rhs):
		CudaData(rhs), 
		width  (rhs.width ),
		height (rhs.height),
		bits(rhs.bits)
	{
		//cout << "cuImgData::ctor" << endl;
	}
	cuImgData(cuImgData&& rhs) noexcept:
		CudaData(std::move(rhs)), 
		width  (std::exchange(rhs.width , 0)),
		height (std::exchange(rhs.height, 0)),
		bits   (std::exchange(rhs.bits  , 0))
	{
		//cout << "cuImgData::cmove" << endl;
	}
	cuImgData& operator=(const cuImgData& rhs) {
		//cout << "cuImgData::copy" << endl;
		if (this != &rhs) {
			resize(rhs);
			CudaData::operator=(rhs);
			width  = rhs.width;
			height = rhs.height;
			bits   = rhs.bits;
		}
		return *this;
	}
	cuImgData& operator=(cuImgData&& rhs) noexcept {
		//cout << "cuImgData::move" << endl;
		if(this != &rhs) {
			this->~cuImgData();
			CudaData::operator=(std::move(rhs));
			width  = std::exchange(rhs.width , 0);
			height = std::exchange(rhs.height, 0);
			bits   = std::exchange(rhs.bits  , 0);
		}
		return *this;
	}
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
		ImgData_resize(dst, width, height, bits);
		memcpyOut(dst.raw_img.data(), dst.raw_img.size());
	}
public:
	void resize(uint32_t width, uint32_t height, uint16_t bits) {
		// 空間不足重new
		if(width*height > this->len) {
			CudaData::resize(width*height * bits>>3);
			this->width  = width;
			this->height = height;
			this->bits   = bits;
			//cout << "reNewSize" << endl;
		} 
		// 空間充足直接用
		else if(width*height <= this->len) {
			//cout << "non reNewSize" << endl;
		}
		this->width  = width;
		this->height = height;
		this->bits   = bits;
	}
	void resize(const cuImgData& src) {
		resize(src.width, src.height, src.bits);
	}
	void resize(const basic_ImgData& src) {
		resize(src.width, src.height, src.bits);
	}
public:
	int size() const {
		return width*height *bits>>3;
	}
	int sizePix() const {
		return width*height;
	}
	void info_print(bool detail=0) const {
		std::cout << ">>IMG basic info:" << "\n";
		std::cout << "  - img size  = " << this->size() << "\n";
		std::cout << "  - img width = " << this->width  << "\n";
		std::cout << "  - img heigh = " << this->height << "\n";
		std::cout << "  - img bits  = " << this->bits   << "\n";
		if(detail) {
			std::cout << "  -  gpu data address  = "  << (int)*gpuData << "\n";
			std::cout << "  -  gpu data capacity  = " << len     << "\n";
		}
		std::cout << "\n";
	}
public:
	uint32_t width  = 0;
	uint32_t height = 0;
	uint16_t bits   = 0;
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



