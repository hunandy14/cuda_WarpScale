/*****************************************************************
Name : 
Date : 2018/01/08
By   : CharlotteHonG
Final: 2018/01/08
*****************************************************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <string>
using namespace std;

#include "cubilinear/cubilinear.hpp"
#include "OpenBMP.hpp"
#include "Timer.hpp"
#include "LapBlend/LapBlend.hpp"

#include <opencv2\opencv.hpp>

using uch = unsigned char;

vector<float> tofloat(const uch* img, size_t size) {
	vector<float> temp(size);
	for(size_t i = 0; i < size; i++) {
		temp[i] = img[i];
	} return temp;
}
vector<uch> touch(const float* img, size_t size) {
	vector<uch> temp(size);
	for(size_t i = 0; i < size; i++) {
		temp[i] = img[i];
	} return temp;
}
void imgSub(basic_ImgData &src, const basic_ImgData &dst);
void cuda_info() {
	cudaDeviceProp prop;  

	int count;  
	cudaGetDeviceCount(&count);
	printf("顯卡所支持的cuda處理器數量：%d\n", count);  

	for (int i = 0; i < count; ++i){  
		cudaGetDeviceProperties(&prop , i);  
		printf("------------第%d個處理器的基本信息------------\n" ,i+1 );  
		printf("處理器名稱：%s \n" , prop.name );  
		printf("計算能力：%d.%d\n" ,prop.major , prop.minor);  
		printf("設備上全局內存總量：%dMB\n" ,prop.totalGlobalMem/1024/1024 );  
		printf("設備上常量內存總量：%dKB\n", prop.totalConstMem/1024);  
		printf("一個線程塊中可使用的最大共享內存：%dKB\n", prop.sharedMemPerBlock / 1024);  
		printf("一個線程束包含的線程數量：%d\n", prop.warpSize);  
		printf("一個線程塊中可包含的最大線程數量：%d\n", prop.maxThreadsPerBlock);  
		printf("多維線程塊數組中每一維可包含的最大線程數量：(%d,%d,%d)\n", prop.maxThreadsDim[0],  
			prop.maxThreadsDim[1], prop.maxThreadsDim[2] );  
		printf("一個線程格中每一維可包含的最大線程塊數量：(%d,%d,%d)\n", prop.maxGridSize[0],  
			prop.maxGridSize[1], prop.maxGridSize[2]);  
	} printf("---------------------------------------------\n");
}
int main(){
	//查看显卡配置
	cuda_info();

	Timer T;
	double ratio = 1;
	// 讀取
	ImgData src("img/_test.bmp"); ratio = 2;
	//ImgData src("img/737400.bmp"); ratio = 1;
	ImgData dst;
	dst.resize(src.width*ratio, src.height*ratio, src.bits);

	// 要求GPU空間
	cuImgData uSrc(src);
	cuImgData uDst;
	uDst.resize(src.width*ratio, src.height*ratio, src.bits);

	// GPU速度
	T.start();
	WarpScale_rgb(uSrc, uDst, ratio);
	T.print(" cuWarpScale_rgb");
	
	// GPU 輸出
	T.start();
	uDst.out(dst);
	T.print(" gpu out data");
	//dst.bmp("cutestImg.bmp");

	// CPU速度
	T.start();
	WarpScale_rgb(src, dst, ratio);
	T.print(" WarpScale_rgb");
	//dst.bmp("testImg.bmp");

	/* 金字塔混和 */
	cout << "\n\n金字塔混和\n" << endl;

	ImgData t1("img/_Test0.bmp"), out;
	cuImgData ut1(t1), ut2;
	ut2.resize(t1);

	imgGau(ut1, ut2);

	ut2.out(out);
	out.bmp("__bugTest.bmp");

	LapBlend_Tester();


	return 0;
}