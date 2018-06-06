/***************************************************************************************
Name :
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
***************************************************************************************/
#include <iostream>
#include <vector>
using namespace std;

#include "bilinear.cuh"
#define BLOCK_DIM 16.0

using uch = unsigned char;


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
__global__ void cuWarpScale_rgb_kernel(const uch* src, uch* dst, 
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
__host__ void cuWarpScale_rgb(const ImgData & src, ImgData & dst, double ratio){
	Timer t;
	// ��l�ƪŶ�
	dst.resize(src.width*ratio, src.height*ratio, src.bits);
	// �n�DGPU�Ŷ�
	t.start();
	CudaData<uch> gpuSrc(src.raw_img.data(), src.size());
	CudaData<uch> gpuDst(dst.size());
	t.print("  cudamalloc");
	// �]�m�����
	dim3 block(BLOCK_DIM, BLOCK_DIM);
	dim3 grid(ceil(dst.width / BLOCK_DIM), ceil(dst.width / BLOCK_DIM));
	// ���� kernel
	t.start();
	cuWarpScale_rgb_kernel <<< grid, block >>> (gpuSrc, gpuDst, src.width, src.height, ratio);
	t.print("  kernel");
	gpuDst.memcpyOut(dst.raw_img.data(), dst.size());
}



//======================================================================================
// �ֳt�u�ʴ���_�֤�
static inline
void fast_Bilinear_rgb(unsigned char* p, 
	const basic_ImgData& src, double y, double x)
{
	// �_�I
	int _x = (int)x;
	int _y = (int)y;
	// ������
	double l_x = x - (double)_x;
	double r_x = 1.f - l_x;
	double t_y = y - (double)_y;
	double b_y = 1.f - t_y;
	int srcW = src.width;
	int srcH = src.height;

	// �p��RGB
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
// �ֳt�u�ʴ���
void WarpScale_rgb(const basic_ImgData &src, basic_ImgData &dst, double ratio){
	// ���b
	if (src.bits != 24) runtime_error("IMG is not 24bit.");
	// ��l�� dst
	dst.width  = (int)((src.width  * ratio) +0.5);
	dst.height = (int)((src.height * ratio) +0.5);
	dst.bits   = src.bits;
	dst.raw_img.resize(dst.width * dst.height * dst.bits>>3);

	// �Y�p�����v
	double r1W = ((double)src.width )/(dst.width );
	double r1H = ((double)src.height)/(dst.height);
	// ��j�����v
	double r2W = (src.width -1.0)/(dst.width -1.0);
	double r2H = (src.height-1.0)/(dst.height-1.0);
	// �Y�p�ɭԪ��~�t
	double deviW = ((src.width-1.0)  - (dst.width -1.0)*(r1W)) /dst.width;
	double deviH = ((src.height-1.0) - (dst.height-1.0)*(r1H)) /dst.height;

	// �]�s�Ϯy��
//#pragma omp parallel for
	for (int j = 0; j < dst.height; ++j) {
		for (int i = 0; i < dst.width; ++i) {
			// �վ���
			double srcY, srcX;
			if (ratio < 1.0) {
				//srcY = ((j+0.5f)/Ratio) - 0.5;
				//srcX = ((i+0.5f)/Ratio) - 0.5;
				srcX = i*(r1W+deviW);
				srcY = j*(r1H+deviH);
			} else if (ratio >= 1.0) {
				srcX = i*r2W;
				srcY = j*r2H;
			}
			// ������ɭ�
			unsigned char* p = &dst.raw_img[(j*dst.width + i) *3];
			fast_Bilinear_rgb(p, src, srcY, srcX);
		}
	}
}


//======================================================================================