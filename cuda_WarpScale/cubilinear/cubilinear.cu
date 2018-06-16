/***************************************************************************************
Name :
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
***************************************************************************************/
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

#include "cubilinear.hpp"
#define BLOCK_DIM 16.0
#define BLOCK_DIM_X 32.0
#define BLOCK_DIM_Y 8.0

using uch = unsigned char;


//======================================================================================
// 快速線性插值_核心
__device__ __host__ static inline
void fast_Bilinear(const uch* src, int w, int h,
	uch* p, double y, double x)
{
	int srcW = w;
	int srcH = h;

	// 起點
	int _x = (int)x;
	int _y = (int)y;
	// 左邊比值
	double l_x = x - (double)_x;
	double r_x = 1.f - l_x;
	double t_y = y - (double)_y;
	double b_y = 1.f - t_y;

	// 計算RGB
	double R , G, B;
	int x2 = (_x+1) > srcW -1? srcW -1: _x+1;
	int y2 = (_y+1) > srcH-1? srcH-1: _y+1;

	R  = (double)src[(_y * srcW + _x) *3 + 0] * (r_x * b_y);
	G  = (double)src[(_y * srcW + _x) *3 + 1] * (r_x * b_y);
	B  = (double)src[(_y * srcW + _x) *3 + 2] * (r_x * b_y);
	R += (double)src[(_y * srcW + x2) *3 + 0] * (l_x * b_y);
	G += (double)src[(_y * srcW + x2) *3 + 1] * (l_x * b_y);
	B += (double)src[(_y * srcW + x2) *3 + 2] * (l_x * b_y);
	R += (double)src[(y2 * srcW + _x) *3 + 0] * (r_x * t_y);
	G += (double)src[(y2 * srcW + _x) *3 + 1] * (r_x * t_y);
	B += (double)src[(y2 * srcW + _x) *3 + 2] * (r_x * t_y);
	R += (double)src[(y2 * srcW + x2) *3 + 0] * (l_x * t_y);
	G += (double)src[(y2 * srcW + x2) *3 + 1] * (l_x * t_y);
	B += (double)src[(y2 * srcW + x2) *3 + 2] * (l_x * t_y);

	p[0] = (unsigned char) R;
	p[1] = (unsigned char) G;
	p[2] = (unsigned char) B;
}
//==============================================
// 快速線性插值
__global__ 
void cuWarpScale_kernel(const uch* src, uch* dst, 
	int w, int h, double ratio)
{
	int srcW=w;
	int srcH=h;

	int dstW = (int)((srcW * ratio) +0.5);
	int dstH = (int)((srcH * ratio) +0.5);

	// 縮小的倍率
	double r1W = ((double)srcW )/(dstW);
	double r1H = ((double)srcH)/(dstH);
	// 放大的倍率
	double r2W = (srcW -1.0)/(dstW -1.0);
	double r2H = (srcH-1.0)/(dstH-1.0);
	// 縮小時候的誤差
	double deviW = ((srcW-1.0)  - (dstW -1.0)*(r1W)) /dstW;
	double deviH = ((srcH-1.0) - (dstH-1.0)*(r1H)) /dstH;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < dstH && i < dstW) { // 會多跑一點點要擋掉
			double srcY, srcX;
			if (ratio < 1.0) {
				srcX = i*(r1W+deviW);
				srcY = j*(r1H+deviH);
			} else if (ratio >= 1.0) {
				srcX = i*r2W;
				srcY = j*r2H;
			}
			// 獲取插補值
			unsigned char* p = &dst[(j*dstW+ i) *3];
			fast_Bilinear(src, w, h, p, srcY, srcX);
		
	}
}
// GPU 線性插值
__host__
void WarpScale_rgb(const cuImgData & uSrc, cuImgData & uDst, double ratio) {
	// 設置大小
	int dstW = (int)((uSrc.width  * ratio) +0.5);
	int dstH = (int)((uSrc.height * ratio) +0.5);
	// 不相同則resize
	if(uDst.width != dstW || uDst.height != dstH || uDst.bits != uSrc.bits) 
		uDst.resize(dstW, dstH, uSrc.bits);
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(dstW / BLOCK_DIM_X), ceil(dstH / BLOCK_DIM_Y));
	// 執行 kernel
	cuWarpScale_kernel <<< grid, block >>> (uSrc, uDst, uSrc.width, uSrc.height, ratio);
}
// CPU快速線性插值
__host__
void WarpScale_rgb(const basic_ImgData &src, basic_ImgData &dst, double ratio){
	// 防呆
	if (src.bits != 24) runtime_error("IMG is not 24bit.");
	// 初始化 dst
	dst.width  = (int)((src.width  * ratio) +0.5);
	dst.height = (int)((src.height * ratio) +0.5);
	dst.bits   = src.bits;
	dst.raw_img.resize(dst.width * dst.height * dst.bits>>3);

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
//#pragma omp parallel for
	for (int j = 0; j < dst.height; ++j) {
		for (int i = 0; i < dst.width; ++i) {
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
			unsigned char* p = &dst.raw_img[(j*dst.width + i) *3];
			fast_Bilinear(src.raw_img.data(), src.width, src.height, p, srcY, srcX);
		}
	}
}


// 測試 cuWarpScale_kernel
__host__
void cuWarpScale_kernel_test(const basic_ImgData & src, basic_ImgData & dst, double ratio){
	Timer t;
	// 初始化空間
	//t.start();
	// 初始化 dst
	dst.width  = (int)((src.width  * ratio) +0.5);
	dst.height = (int)((src.height * ratio) +0.5);
	dst.bits   = src.bits;
	dst.raw_img.resize(dst.width * dst.height * dst.bits>>3);
	//t.print("  resize");

	// 要求GPU空間
	//t.start();
	CudaData<uch> gpuSrc(src.raw_img.size());
	//t.print("  cudamalloc gpuSrc");
	//t.start();
	CudaData<uch> gpuDst(dst.raw_img.size());
	//t.print("  cudamalloc gpuDst");

	// 複製資料
	// t.start();/
	gpuSrc.memcpyIn(src.raw_img.data(), src.raw_img.size());
	// t.print("  memcpyIn");

	// 設置執行緒
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil(dst.width / BLOCK_DIM), ceil(dst.width / BLOCK_DIM));

	// 執行 kernel
	// t.start();
	cuWarpScale_kernel <<< grid, block >>> (gpuSrc, gpuDst, src.width, src.height, ratio);
	// t.print("  kernel");

	// 複製資料
	// t.start();
	gpuDst.memcpyOut(dst.raw_img.data(), dst.raw_img.size());
	// t.print("  memcpyOut");


	// t.start();
	gpuDst.~CudaData<uch>();
	// t.print("  dctor");
}


//======================================================================================
// 圖像相減
__global__
void imgSub_kernel(uch* src, int srcW, int srcH, const uch* dst, int dstW, int dstH)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < srcH && i < srcW) { // 會多跑一點點要擋掉
		int srcIdx = (j*srcW + i) * 3;
		int dstIdx = (j*dstW + i) * 3;

		int pixR = (int)src[srcIdx+0] - (int)dst[dstIdx+0] +128;
		int pixG = (int)src[srcIdx+1] - (int)dst[dstIdx+1] +128;
		int pixB = (int)src[srcIdx+2] - (int)dst[dstIdx+2] +128;

		pixR = pixR <0? 0: pixR;
		pixG = pixG <0? 0: pixG;
		pixB = pixB <0? 0: pixB;
		pixR = pixR >255? 255: pixR;
		pixG = pixG >255? 255: pixG;
		pixB = pixB >255? 255: pixB;

		src[srcIdx+0] = pixR;
		src[srcIdx+1] = pixG;
		src[srcIdx+2] = pixB;
	}
}
// 圖像相減
__host__
void imgSub(cuImgData & uSrc, const cuImgData & uDst) {
	// 設置大小
	int srcW = uSrc.width;
	int srcH = uSrc.height;
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(srcW / BLOCK_DIM_X)+1, ceil(srcH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	imgSub_kernel <<< grid, block >>> (uSrc, uSrc.width, uSrc.height, uDst, uDst.width, uDst.height);
}


//======================================================================================
// 高斯核心
static vector<double> getGaussianKernel( int n, double sigma)
{
	const int SMALL_GAUSSIAN_SIZE = 7;
	static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
	{
		{1.f},
		{0.25f, 0.5f, 0.25f},
		{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
		{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
	};

	const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
		small_gaussian_tab[n>>1] : 0;

	vector<double> kernel(n);
	double* cd = kernel.data();

	double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
	double scale2X = -0.5/(sigmaX*sigmaX);
	double sum = 0;

	int i;
	for( i = 0; i < n; i++ )
	{
		double x = i - (n-1)*0.5;
		double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
		cd[i] = t;
		sum += cd[i];
	}

	sum = 1./sum;

	for( i = 0; i < n; i++ )
	{
		cd[i] *= sum;
	}

	return kernel;
}
//==============================================
// 高私模糊
__global__
void imgGau_kernel(
	const uch* src, int srcW, int srcH, 
	uch* dst, uch* img_gauX,
	const double* gauMat, int matLen)
{
	const int r = matLen / 2;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if( j>=0 && i >=0 &&
		j < srcH && i < srcW) { // 會多跑一點點要擋掉
		int srcIdx = (j*srcW + i) * 3;
		int dstIdx = (j*srcW + i) * 3;

		double sumR = 0;
		double sumG = 0;
		double sumB = 0;

		for (int k = 0; k < matLen; ++k) {
			int idx = i-r + k;
			// idx超出邊緣處理
			if (idx < 0) {
				idx = 0;
			} else if (idx >(int)(srcW-1)) {
				idx = (srcW-1);
			}
			sumR += (double)src[(j*srcW + idx)*3 + 0] * gauMat[k];
			sumG += (double)src[(j*srcW + idx)*3 + 1] * gauMat[k];
			sumB += (double)src[(j*srcW + idx)*3 + 2] * gauMat[k];
		}

		dst[srcIdx+0] = sumR;
		dst[srcIdx+1] = sumG;
		dst[srcIdx+2] = sumB;
	}
}
__host__
void GaussianBlur(const cuImgData& uSrc, cuImgData & uDst, int matLen, double sigma) {
	// 設置大小
	int srcW = uSrc.width;
	int srcH = uSrc.height;
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(srcW / BLOCK_DIM_X)+1, ceil(srcH / BLOCK_DIM_Y)+1);
	// 要求GPU空間
	cuImgData uTemp;
	uDst.resize(uSrc);
	uTemp.resize(uSrc);
	// 高斯 kernle
	vector<double> gau_mat = getGaussianKernel(matLen, sigma);
	CudaData<double> uMat(gau_mat.data(), gau_mat.size());
	// 執行 kernel
	imgGau_kernel <<< grid, block >>> (
		uSrc, uSrc.width, uSrc.height, 
		uDst, uTemp,
		uMat, matLen
	);
}

