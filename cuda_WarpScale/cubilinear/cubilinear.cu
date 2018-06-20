/***************************************************************************************
Name :
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
***************************************************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <cmath>
using namespace std;

#include "cubilinear.hpp"
#define BLOCK_DIM 16.0
#define BLOCK_DIM_X 32.0
#define BLOCK_DIM_Y 8.0

using uch = unsigned char;

//======================================================================================
// 複製圖片
__global__
void imgCopy_kernel(const uch* src, int srcW, int srcH, uch* dst, int dstW, int dstH)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < srcH && i < srcW) { // 會多跑一點點要擋掉
		int srcIdx = (j*srcW +i) *3;
		int dstIdx = (j*dstW +i) *3;

		dst[dstIdx+0] = src[srcIdx+0];
		dst[dstIdx+1] = src[srcIdx+1];
		dst[dstIdx+2] = src[srcIdx+2];
	}					
}
// 複製圖片
__host__
void imgCopy(const cuImgData & uSrc, cuImgData & uDst) {
	// 設置大小
	int srcW = uSrc.width;
	int srcH = uSrc.height;

	uDst.resize(uSrc);

	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(srcW / BLOCK_DIM_X)+1, ceil(srcH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	imgCopy_kernel <<< grid, block >>> (uSrc, uSrc.width, uSrc.height, uDst, uDst.width, uDst.height);
}


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
// 圖像相加
__global__
void imgAdd_kernel(uch* src, int srcW, int srcH, const uch* dst, int dstW, int dstH)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < srcH && i < srcW) { // 會多跑一點點要擋掉
		int srcIdx = (j*srcW + i) * 3;
		int dstIdx = (j*dstW + i) * 3;

		int pixR = (int)src[srcIdx+0] + (int)dst[dstIdx+0] -128;
		int pixG = (int)src[srcIdx+1] + (int)dst[dstIdx+1] -128;
		int pixB = (int)src[srcIdx+2] + (int)dst[dstIdx+2] -128;

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
__host__
void imgAdd(cuImgData & uSrc, const cuImgData & uDst) {
	// 設置大小
	int srcW = uSrc.width;
	int srcH = uSrc.height;
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(srcW / BLOCK_DIM_X)+1, ceil(srcH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	imgAdd_kernel <<< grid, block >>> (uSrc, uSrc.width, uSrc.height, uDst, uDst.width, uDst.height);
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
// 高私模糊核心
__global__
void imgGauX_kernel( const uch* src, int srcW, int srcH, 
	uch* dst, uch* img_gauX, const double* gauMat, int matLen)
{
	const int r = matLen / 2;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if( j>=0 && i >=0 &&
		j < srcH && i < srcW) { // 會多跑一點點要擋掉
		int srcIdx = (j*srcW + i) * 3;

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
// 高斯模糊
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
	// 執行 kernel (單X方向)
	imgGauX_kernel <<< grid, block >>> (
		uSrc, uSrc.width, uSrc.height, 
		uDst, uTemp,
		uMat, matLen
	);
}



//======================================================================================
// 混合
__global__
void imgBlendHalf_kernel(const uch* imgA, const uch* imgB, uch* dst, int dstW, int dstH)
{
	int center = dstW >>1;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < dstH && i < dstW) { // 會多跑一點點要擋掉
		int dstIdx = (j* dstW+i)*3;
		int LAIdx  = (j*dstW+i)*3;
		int LBIdx  = (j*dstW+i)*3;

		for(int rgb = 0; rgb < 3; rgb++) {
			// 拉普拉斯差值區 (左邊就放左邊差值，右邊放右邊差值，正中間放平均)
			if(i == center) {// 正中間
				dst[dstIdx +rgb] = (imgA[LAIdx +rgb] + imgB[LBIdx +rgb]) >>1;
			} else if(i > center) {// 右半部
				dst[dstIdx +rgb] = imgB[LBIdx +rgb];
			} else { // 左半部
				dst[dstIdx +rgb] = imgA[LAIdx +rgb];
			}
		}
	}
}
__host__
void imgBlendHalf(const cuImgData& uimgA, const cuImgData& uimgB, cuImgData& uDst) {
	uDst.resize(uimgA);

	// 設置大小
	int dstW = uDst.width;
	int dstH = uDst.height;
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(dstW / BLOCK_DIM_X)+1, ceil(dstH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	imgBlendHalf_kernel <<< grid, block >>> (uimgA, uimgB, uDst, uDst.width, uDst.height);
}
__global__
void imgBlendAlpha_kernel(const uch* imgA, const uch* imgB, uch* dst, int dstW, int dstH)
{
	double rat = 1.0 / dstW;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < dstH && i < dstW) { // 會多跑一點點要擋掉
		int dstIdx = (j* dstW+i)*3;
		int LAIdx  = (j*dstW+i)*3;
		int LBIdx  = (j*dstW+i)*3;

		for(int rgb = 0; rgb < 3; rgb++) {
			double r1 = rat*i;
			double r2 = 1.0-r1;
			dst[dstIdx +rgb] = imgA[LAIdx +rgb]*r2 + imgB[LBIdx +rgb]*r1;
		}
	}
}
__host__
void imgBlendAlpha(const cuImgData& uimgA, const cuImgData& uimgB, cuImgData& uDst) {
	uDst.resize(uimgA);

	// 設置大小
	int dstW = uDst.width;
	int dstH = uDst.height;
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(dstW / BLOCK_DIM_X)+1, ceil(dstH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	imgBlendAlpha_kernel <<< grid, block >>> (uimgA, uimgB, uDst, uDst.width, uDst.height);
}


//======================================================================================
// 合併圖片
__global__
void mergeOverlap_kernel(
	const uch* usrc1, int src1W, int src1H,
	const uch* usrc2, int src2W, int src2H,
	const uch* ublend, int blendW, int blendH,
	uch* udst, int dstW, int dstH,
	int* ucorner)
{
	// 偏移量
	int mx = ucorner[4];
	int my = ucorner[5];
	// 兩張圖的高度偏差值
	int myA = my<0? 0:my;
	int myB = my>0? 0:-my;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < dstH && i < dstW) { // 會多跑一點點要擋掉
		int dstIdx = (j*dstW+ i) *3;
		// 圖1
		if(i < mx) {
			for(int rgb = 0; rgb < 3; rgb++) {
				udst[dstIdx +rgb] = usrc1[(((j+myA)+ucorner[1])*src1W +(i+ucorner[0])) *3+rgb];
			}
		}
		// 重疊區
		else if(i >= mx && i < ucorner[2]-ucorner[0]) {
			for(int rgb = 0; rgb < 3; rgb++) {
				udst[dstIdx +rgb] = ublend[(j*blendW+(i-mx)) *3+rgb];
			}
		}
		// 圖2
		else if(i >= ucorner[2]-ucorner[0]) {
			for(int rgb = 0; rgb < 3; rgb++) {
				udst[dstIdx +rgb] = usrc2[(((j+myB)+ucorner[1])*src2W +((i-mx)+ucorner[0])) *3+rgb];
			}
		}
	}					
}
__host__
void mergeOverlap(const cuImgData& uSrc, const cuImgData& uSrc2,
	const cuImgData& uBlend, cuImgData& uDst, vector<int> corner)
{
	CudaData<int> ucorner(corner.data(), corner.size());
	
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 兩張圖疊起來大小
	int newH=corner[3]-corner[1]-abs(my);
	int newW=corner[2]-corner[0]+mx;
	uDst.resize(newW, newH, uSrc.bits);

	// 設置大小
	uDst.resize(newW, newH, uSrc.bits);

	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(newW / BLOCK_DIM_X)+1, ceil(newH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	mergeOverlap_kernel <<< grid, block >>> (
		uSrc, uSrc.width, uSrc.height,
		uSrc2, uSrc2.width, uSrc2.height,
		uBlend, uBlend.width, uBlend.height,
		uDst, uDst.width, uDst.height,
		ucorner
	);
}


//======================================================================================
// 重疊區
__global__
void getOverlap_kernel2(
	const uch* src1, int src1W, int src1H,
	const uch* src2, int src2W, int src2H,
	uch* cut1, int cut1W, int cut1H,
	uch* cut2, int cut2W, int cut2H,
	int* corner)
{
	// 偏移量
	const int mx=corner[4];
	const int my=corner[5];
	// 重疊區大小
	const int lapH=corner[3]-corner[1]-abs(my);
	const int lapW=corner[2]-corner[0]-mx;
	// 兩張圖的高度偏差值
	const int myA = my<0? 0:my;
	const int myB = my>0? 0:-my;

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(j < lapH && i < lapW) { // 會多跑一點點要擋掉
		if (i < corner[2]-corner[0]-mx) {
			for (int  rgb = 0; rgb < 3; rgb++) {
				// 圖1
				cut1[(j*cut2W +(i)) *3+rgb] = 
					src1[(((j+myA)+corner[1])*src1W +(i+corner[0]+mx)) *3+rgb];
				// 圖2
				cut2[(j*cut2W +(i)) *3+rgb] = 
					src2[(((j+myB)+corner[1])*src2W +((i)+corner[0])) *3+rgb];
			}
		}
	}					
}
__host__
void getOverlap(const cuImgData& uSrc, const cuImgData& uSrc2,
	cuImgData& ucut1, cuImgData& ucut2, vector<int> corner)
{
	CudaData<int> ucorner(corner.data(), corner.size());
	//cout << ucorner[0] << endl;

	// 偏移量
	const int mx=corner[4];
	const int my=corner[5];
	// 重疊區大小
	const int lapH=corner[3]-corner[1]-abs(my);
	const int lapW=corner[2]-corner[0]-mx;

	// 設置大小
	ucut1.resize(lapW, lapH, uSrc.bits);
	ucut2.resize(lapW, lapH, uSrc2.bits);
	
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(lapW / BLOCK_DIM_X)+1, ceil(lapH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	getOverlap_kernel2 <<< grid, block >>> (
		uSrc, uSrc.width, uSrc.height,
		uSrc2, uSrc2.width, uSrc2.height,
		ucut1, ucut1.width, ucut1.height,
		ucut2, ucut2.width, ucut2.height,
		ucorner
	);
}


//======================================================================================
// 圓柱投影座標反轉換
__device__ __host__ inline static
void WarpCylindrical_CoorTranfer_Inve(double R,
	size_t width, size_t height, double& x, double& y)
{
	double r2 = (x - width*.5);
	double k = sqrt(R*R + r2*r2) / R;
	x = (x - width *.5)*k + width *.5;
	y = (y - height*.5)*k + height*.5;
}

// 圓柱投影
__global__
void WarpCylindrical_kernel(
	const uch* src, int srcW, int srcH, 
	uch* dst, int dstW, int dstH, 
	double R ,int mx=0, int my=0, double edge=0.0
)
{
	// 設置位移
	int moveH = (srcH*edge) + my;
	int moveW = mx;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < srcH && i < srcW) { // 會多跑一點點要擋掉
		double x = i, y = j;
		WarpCylindrical_CoorTranfer_Inve(R, srcW, srcH, x, y);
		if (x >= 0 && y >= 0 && x < srcW - 1 && y < srcH - 1) {
			unsigned char* p = &dst[((j+moveH)*(srcW+moveW) + (i+moveW)) *3];
			fast_Bilinear(src, srcW, srcH, p, y, x);
		}
	}					
}
__host__
void WarpCylindrical(const cuImgData & uSrc, cuImgData & uDst, 
	double R ,int mx, int my, double edge)
{
	int srcW = uSrc.width;
	int srcH = uSrc.height;
	int dstW = srcW+mx;
	int dstH = srcH * (1+edge*2);

	// 設置大小
	uDst.resize(dstW, dstH, uSrc.bits);

	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(srcW / BLOCK_DIM_X)+1, ceil(srcH / BLOCK_DIM_Y)+1);
	// 執行 kernel
	WarpCylindrical_kernel <<< grid, block >>> (
		uSrc, uSrc.width, uSrc.height, 
		uDst, uDst.width, uDst.height, 
		R, mx, my, edge
	);
}


//======================================================================================
// 圓柱投影邊緣角點
__global__
void WarpCyliCorner_kernel(
	const uch* src, int srcW, int srcH, 
	int* corner, int mx, int my)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(j < 1 && i == 0) {
		// 設置偏移位置
		corner[4]=mx, corner[5]=my;
		// 左上角角點
		for (int i = 0; i < srcW; i++) {
			int pix = (int)src[(srcH/2*srcW +i)*3 +0];
			if (i < (srcW>>1) && pix != 0) {
				corner[0]=i;
				//cout << "corner=" << corner[0] << endl;
				i = srcW>>1;
			} else if (i > (srcW>>1) && pix == 0) {
				corner[2] = i-1;
				//cout << "corner=" << corner[2] << endl;
				break;
			}
		}
	}
	if(j < 1 && i == 1) {
		// 右上角角點
		for (int j = 0; j < srcH; j++) {
			int pix = (int)src[(j*srcW +corner[0])*3 +0];
			if (j < (srcH>>1) && pix != 0) {
				corner[1] = j;
				//cout << "corner=" << corner[2] << endl;
				j = srcH>>1;
			} else if (j > (srcH>>1) && pix == 0) {
				corner[3] = j-1;
				//cout << "corner=" << corner[3] << endl;
				break;
			}
		}
	}					
}
__host__
void WarpCyliCorner(const cuImgData & uSrc, CudaData<int>& ucorner, int mx, int my) {
	// 設置大小
	int newW = 2;
	int newH = 1;
	// 重設大小
	if(ucorner.len<6) {
		ucorner.~CudaData();
		ucorner.malloc(6);
	}
	// 設置執行緒
	dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 grid(ceil(newW / BLOCK_DIM_X)+1, ceil(newH / BLOCK_DIM_Y)+1);

	// 執行 kernel
	WarpCyliCorner_kernel <<< grid, block >>> (
		uSrc, uSrc.width, uSrc.height, 
		ucorner, mx, my
	);
}


//======================================================================================
