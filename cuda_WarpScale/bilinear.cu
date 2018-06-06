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

#define BLOCK_DIM 16.0

using uch = unsigned char;

__host__ __device__
inline static float bilinearRead(const float* img, 
	size_t width, float y, float x) // �u�ʨ���
{
	// ����F�I(����� 1+)
	size_t x0 = floor(x);
	size_t x1 = ceil(x);
	size_t y0 = floor(y);
	size_t y1 = ceil(y);
	// ������(�u��� 1-)
	float dx1 = x - x0;
	float dx2 = 1 - dx1;
	float dy1 = y - y0;
	float dy2 = 1 - dy1;
	// ����I
	const float& A = img[y0*width + x0];
	const float& B = img[y0*width + x1];
	const float& C = img[y1*width + x0];
	const float& D = img[y1*width + x1];
	// ���X���(�n��e)
	float AB = A*dx2 + B*dx1;
	float CD = C*dx2 + D*dx1;
	float X = AB*dy2 + CD*dy1;
	return X;
}
//======================================================================================
// �@�ɰO����u�ʨ��Ȯ֤�
__global__ void biliner_share_kernel(float* dst, const float* src, int srcW, int srcH, float ratio) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int newH = (int)(floor(srcH * ratio));
	int newW = (int)(floor(srcW * ratio));
	if(i < srcW*ratio && j < srcH*ratio) { // �|�h�]�@�I�I�n�ױ�
		// �վ���
		float srcY, srcX;
		if (ratio < 1) {
			srcY = ((j+0.5f)/ratio) - 0.5;
			srcX = ((i+0.5f)/ratio) - 0.5;
		} else {
			srcY = j * (srcH-1.f) / (newH-1.f);
			srcX = i * (srcW -1.f) / (newW-1.f);
		}
		// ������ɭ�
		dst[j*newW + i] = bilinearRead(src, srcW, srcY, srcX);
	}
}
// �@�ɰO����u�ʨ��Ȩ禡
__host__ void biliner_share_core(float *dst, const float* src,
	size_t srcW, size_t srcH, float ratio)
{
	Timer T; T.priSta = 1;
	// �]�mGPU�һݪ���
	int srcSize = srcW*srcH;
	int dstSize = srcSize*ratio*ratio;

	// �n�DGPU�Ŷ�
	T.start();
	CudaData<float> gpu_src(srcSize);
	T.print("  GPU new �Ŷ�1");
	T.start();
	CudaData<float> gpu_dst(dstSize);
	T.print("  GPU new �Ŷ�2");
	// �ƻs��GPU
	T.start();
	gpu_src.memcpyIn(src, srcSize);
	T.print("  GPU �ƻs");

	// �]�m�����
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil((float)srcW*ratio / BLOCK_DIM), ceil((float)srcH*ratio / BLOCK_DIM));
	T.start();
	biliner_share_kernel <<< grid, block >> > (gpu_dst, gpu_src, srcW, srcH, ratio);
	T.print("  �֤߭p��");

	// ���XGPU��
	T.start();
	gpu_dst.memcpyOut(dst, dstSize);
	T.print("  GPU ���X���");

	// ����GPU�Ŷ�
	T.start();
	gpu_src.~CudaData();
	gpu_dst.~CudaData();
	T.print("  GPU ����Ŷ�");
}

// �@�ɰO����u�ʨ��Ȩ禡 vector �श����
__host__ double biliner_share(vector<float>& dst, const vector<float>& src,
	size_t width, size_t height, float ratio)
{
	Timer T; T.priSta = 1;
	T.start();
	dst.resize(width*ratio * height*ratio);
	T.print(" CPU new �x�s�Ŷ�");
	T.start();
	biliner_share_core(dst.data(), src.data(), width, height, ratio);
	T.print(" GPU ����");
	return T;
}

//======================================================================================
__host__ void biliner_CPU_core(vector<float>& img, const vector<float>& img_ori, 
	size_t width, size_t height, float Ratio)
{
	int newH = static_cast<int>(floor(height * Ratio));
	int newW = static_cast<int>(floor(width  * Ratio));
	img.resize(newH*newW);
	// �]�s�Ϯy��
	for (int j = 0; j < newH; ++j) {
		for (int i = 0; i < newW; ++i) {
			// �վ���
			float srcY, srcX;
			if (Ratio < 1) {
				srcY = ((j+0.5f)/Ratio) - 0.5;
				srcX = ((i+0.5f)/Ratio) - 0.5;
			} else {
				srcY = j * (height-1.f) / (newH-1.f);
				srcX = i * (width -1.f) / (newW-1.f);
			}
			// ������ɭ�
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
	T.print(" CPU new �x�s�Ŷ�");
	T.start();
	biliner_CPU_core(dst, src, width, height, ratio);
	T.print(" CPU ����");
	return T;
}

//======================================================================================
__global__ void copy_kernel(uch* dst, const uch* src, int srcW, int srcH) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i < srcW && j < srcH) {
		dst[j*srcW + i] = src[j*srcW + i];
	}
}

void cutest() {
	// �n�DGPU�Ŷ�
	CudaData<uch> gpu_src(9);
	CudaData<uch> gpu_dst(9);

	vector<uch> arr(9), arr2(9);
	for(size_t i = 0; i < 9; i++) {
		arr[i] = i;
	}
	gpu_src.memcpyIn(arr.data(), 9);

	// �]�m�����
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(3, 3);
	copy_kernel <<< grid, block >> > (gpu_dst, gpu_src, 3, 3);

	gpu_dst.memcpyOut(arr2.data(), 9);
	for(size_t i = 0; i < 9; i++) {
		cout << (int)arr2[i] << endl;
	}
}
//======================================================================================


// �ֳt�u�ʴ���_�֤�
__device__ __host__ static inline
void cufast_Bilinear_rgb(unsigned char* p, 
	const unsigned char* src, int w, int h, double y, double x)
{
	int srcW = w;
	int srcH = h;

	// �_�I
	int _x = (int)x;
	int _y = (int)y;
	// ������
	double l_x = x - (double)_x;
	double r_x = 1.f - l_x;
	double t_y = y - (double)_y;
	double b_y = 1.f - t_y;

	// �p��RGB
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
// �ֳt�u�ʴ���
__global__ void cuWarpScale_rgb(const uch* src, uch* dst, 
	int w, int h, double ratio)
{
	int srcH=h;
	int srcW=w;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int dstH = (int)(floor(srcH * ratio));
	int dstW = (int)(floor(srcW * ratio));

	// �Y�p�����v
	double r1W = ((double)srcW )/(dstW );
	double r1H = ((double)srcH)/(dstH);
	// ��j�����v
	double r2W = (srcW -1.0)/(dstW -1.0);
	double r2H = (srcH-1.0)/(dstH-1.0);
	// �Y�p�ɭԪ��~�t
	double deviW = ((srcW-1.0)  - (dstW -1.0)*(r1W)) /dstW;
	double deviH = ((srcH-1.0) - (dstH-1.0)*(r1H)) /dstH;

	if(i < srcW*ratio && j < srcH*ratio) { // �|�h�]�@�I�I�n�ױ�
		double srcY, srcX;
		if (ratio < 1.0) {
			srcX = i*(r1W+deviW);
			srcY = j*(r1H+deviH);
		} else if (ratio >= 1.0) {
			srcX = i*r2W;
			srcY = j*r2H;
		}
		// ������ɭ�
		unsigned char* p = &dst[(j*dstW+ i) *3];
		cufast_Bilinear_rgb(p, src, srcW, srcH, srcY, srcX);
	}
}
__host__ void WarpScale_rgb(const ImgData & src, ImgData & dst, double ratio){
	// ��l�ƪŶ�
	dst.resize(src.width*ratio, src.height*ratio, src.bits);
	// �n�DGPU�Ŷ�
	CudaData<uch> gpuSrc(src.raw_img.data(), src.size());
	CudaData<uch> gpuDst(dst.size());
	// �]�m�����
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil(dst.width / BLOCK_DIM), ceil(dst.width / BLOCK_DIM));
	// ���� kernel
	cuWarpScale_rgb <<< grid, block >>> (gpuSrc, gpuDst, src.width, src.height, ratio);
	gpuDst.memcpyOut(dst.raw_img.data(), dst.size());
}