/***************************************************************************************
Name :
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
***************************************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
using namespace std;

#include "CudaMem\CudaMem.cuh"
#include "Timer.hpp"
#include "bilinear.cuh"

#define BLOCK_DIM 16

using uch = unsigned char;

__host__ __device__
inline static float bilinearRead(const float* img, 
	size_t width, float y, float x) // 線性取值
{
	// 獲取鄰點(不能用 1+)
	size_t x0 = floor(x);
	size_t x1 = ceil(x);
	size_t y0 = floor(y);
	size_t y1 = ceil(y);
	// 獲取比例(只能用 1-)
	float dx1 = x - x0;
	float dx2 = 1 - dx1;
	float dy1 = y - y0;
	float dy2 = 1 - dy1;
	// 獲取點
	const float& A = img[y0*width + x0];
	const float& B = img[y0*width + x1];
	const float& C = img[y1*width + x0];
	const float& D = img[y1*width + x1];
	// 乘出比例(要交叉)
	float AB = A*dx2 + B*dx1;
	float CD = C*dx2 + D*dx1;
	float X = AB*dy2 + CD*dy1;
	return X;
}
//======================================================================================
// 共享記憶體線性取值核心
__global__ void biliner_share_kernel(float* dst, const float* src, int srcW, int srcH, float ratio) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int newH = (int)(floor(srcH * ratio));
	int newW = (int)(floor(srcW * ratio));
	if(i < srcW*ratio && j < srcH*ratio) { // 會多跑一點點要擋掉
		// 調整對齊
		float srcY, srcX;
		if (ratio < 1) {
			srcY = ((j+0.5f)/ratio) - 0.5;
			srcX = ((i+0.5f)/ratio) - 0.5;
		} else {
			srcY = j * (srcH-1.f) / (newH-1.f);
			srcX = i * (srcW -1.f) / (newW-1.f);
		}
		// 獲取插補值
		dst[j*newW + i] = bilinearRead(src, srcW, srcY, srcX);
	}
}
// 共享記憶體線性取值函式
__host__ void biliner_share_core(float *dst, const float* src,
	size_t srcW, size_t srcH, float ratio)
{
	Timer T; T.priSta = 1;
	// 設置GPU所需長度
	int srcSize = srcW*srcH;
	int dstSize = srcSize*ratio*ratio;

	// 要求GPU空間
	T.start();
	CudaData<float> gpu_src(srcSize);
	T.print("  GPU new 空間1");
	T.start();
	CudaData<float> gpu_dst(dstSize);
	T.print("  GPU new 空間2");
	// 複製到GPU
	T.start();
	gpu_src.memcpyIn(src, srcSize);
	T.print("  GPU 複製");

	// 設置執行緒
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil((float)srcW*ratio / BLOCK_DIM), ceil((float)srcH*ratio / BLOCK_DIM));
	T.start();
	biliner_share_kernel <<< grid, block >> > (gpu_dst, gpu_src, srcW, srcH, ratio);
	T.print("  核心計算");

	// 取出GPU值
	T.start();
	gpu_dst.memcpyOut(dst, dstSize);
	T.print("  GPU 取出資料");

	// 釋放GPU空間
	T.start();
	gpu_src.~CudaData();
	gpu_dst.~CudaData();
	T.print("  GPU 釋放空間");
}

// 共享記憶體線性取值函式 vector 轉介介面
__host__ double biliner_share(vector<float>& dst, const vector<float>& src,
	size_t width, size_t height, float ratio)
{
	Timer T; T.priSta = 1;
	T.start();
	dst.resize(width*ratio * height*ratio);
	T.print(" CPU new 儲存空間");
	T.start();
	biliner_share_core(dst.data(), src.data(), width, height, ratio);
	T.print(" GPU 全部");
	return T;
}



//======================================================================================
__host__ void biliner_CPU_core(vector<float>& img, const vector<float>& img_ori, 
	size_t width, size_t height, float Ratio)
{
	int newH = static_cast<int>(floor(height * Ratio));
	int newW = static_cast<int>(floor(width  * Ratio));
	img.resize(newH*newW);
	// 跑新圖座標
	for (int j = 0; j < newH; ++j) {
		for (int i = 0; i < newW; ++i) {
			// 調整對齊
			float srcY, srcX;
			if (Ratio < 1) {
				srcY = ((j+0.5f)/Ratio) - 0.5;
				srcX = ((i+0.5f)/Ratio) - 0.5;
			} else {
				srcY = j * (height-1.f) / (newH-1.f);
				srcX = i * (width -1.f) / (newW-1.f);
			}
			// 獲取插補值
			img[j*newW + i] = bilinearRead(img_ori.data(), width, srcY, srcX);
		}
	}
}
__host__ double biliner_CPU(vector<float>& dst, const vector<float>& src,
	size_t width, size_t height, float ratio)
{
	Timer T; T.priSta = 1;
	T.start();
	dst.resize(width*ratio * height*ratio);
	T.print(" CPU new 儲存空間");
	T.start();
	biliner_CPU_core(dst, src, width, height, ratio);
	T.print(" CPU 全部");
	return T;
}


//======================================================================================
// 快速線性插值_核心
static inline
void cufast_Bilinear_rgb(unsigned char* p, 
	const cubasic_ImgData& src, double y, double x)
{
	// 起點
	int _x = (int)x;
	int _y = (int)y;
	// 左邊比值
	double l_x = x - (double)_x;
	double r_x = 1.f - l_x;
	double t_y = y - (double)_y;
	double b_y = 1.f - t_y;
	int srcW = src.width;
	int srcH = src.height;

	// 計算RGB
	double R , G, B;
	int x2 = (_x+1) > src.width -1? src.width -1: _x+1;
	int y2 = (_y+1) > src.height-1? src.height-1: _y+1;
	R  = (double)src.raw_img[(_y * srcW + _x) *3 + 0] * (r_x * b_y);
	G  = (double)src.raw_img[(_y * srcW + _x) *3 + 1] * (r_x * b_y);
	B  = (double)src.raw_img[(_y * srcW + _x) *3 + 2] * (r_x * b_y);
	R += (double)src.raw_img[(_y * srcW + x2) *3 + 0] * (l_x * b_y);
	G += (double)src.raw_img[(_y * srcW + x2) *3 + 1] * (l_x * b_y);
	B += (double)src.raw_img[(_y * srcW + x2) *3 + 2] * (l_x * b_y);
	R += (double)src.raw_img[(y2 * srcW + _x) *3 + 0] * (r_x * t_y);
	G += (double)src.raw_img[(y2 * srcW + _x) *3 + 1] * (r_x * t_y);
	B += (double)src.raw_img[(y2 * srcW + _x) *3 + 2] * (r_x * t_y);
	R += (double)src.raw_img[(y2 * srcW + x2) *3 + 0] * (l_x * t_y);
	G += (double)src.raw_img[(y2 * srcW + x2) *3 + 1] * (l_x * t_y);
	B += (double)src.raw_img[(y2 * srcW + x2) *3 + 2] * (l_x * t_y);

	p[0] = (unsigned char) R;
	p[1] = (unsigned char) G;
	p[2] = (unsigned char) B;
}
// 快速線性插值
__global__ void cuWarpScale_rgb(const cubasic_ImgData &src, cubasic_ImgData &dst, double ratio){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	// 初始化 dst
	dst.width  = (int)((src.width  * ratio) +0.5);
	dst.height = (int)((src.height * ratio) +0.5);
	dst.bits   = src.bits;

	int srcH=src.height;
	int srcW=src.width;
	//dst.raw_img.resize(dst.width * dst.height * dst.bits>>3);
	int newH = (int)(floor(srcH * ratio));
	int newW = (int)(floor(srcW * ratio));

	// 縮小的倍率
	double r1W = ((double)src.width )/(dst.width );
	double r1H = ((double)src.height)/(dst.height);
	// 放大的倍率
	double r2W = (src.width -1.0)/(dst.width -1.0);
	double r2H = (src.height-1.0)/(dst.height-1.0);
	// 縮小時候的誤差
	double deviW = ((src.width-1.0)  - (dst.width -1.0)*(r1W)) /dst.width;
	double deviH = ((src.height-1.0) - (dst.height-1.0)*(r1H)) /dst.height;

	// 跑新圖座標
	if(i < srcW*ratio && j < srcH*ratio) {
		// 調整對齊
		double srcY, srcX;
		if (ratio < 1.0) {
			srcX = i*(r1W+deviW);
			srcY = j*(r1H+deviH);
		} else if (ratio >= 1.0) {
			srcX = i*r2W;
			srcY = j*r2H;
		}

		// 獲取插補值
		//unsigned char* p = &dst.raw_img[(j*dst.width + i) *3];
		if (ratio>1) {
			//cufast_Bilinear_rgb(p, src, srcY, srcX);
		} else {
			//cufast_Bilinear_rgb(p, src, srcY, srcX);
		}
	}
}


__host__ void WarpScale_rgb(const basic_ImgData & src, basic_ImgData & dst, double ratio){

}