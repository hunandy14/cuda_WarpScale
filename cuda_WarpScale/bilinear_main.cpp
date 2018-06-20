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

#include "OpenBMP.hpp"
#include "Timer.hpp"

#include "cubilinear/cubilinear.hpp"
#include "LapBlend/LapBlend.hpp"

vector<float> tofloat(const unsigned char* img, size_t size) {
	vector<float> temp(size);
	for(size_t i = 0; i < size; i++) {
		temp[i] = img[i];
	} return temp;
}
vector<unsigned char> touch(const float* img, size_t size) {
	vector<unsigned char> temp(size);
	for(size_t i = 0; i < size; i++) {
		temp[i] = img[i];
	} return temp;
}
void bilinear_test() {
	Timer T;
	double ratio = 1;

	// 讀取
	ImgData src("img/kanna.bmp"); ratio = 2;
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
}

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
		printf("設備上全局內存總量：%d MB\n" ,(int)prop.totalGlobalMem/1024/1024 );
		printf("設備上常量內存總量：%d KB\n", (int)prop.totalConstMem/1024);
		printf("一個線程塊中可使用的最大共享內存：%d KB\n", (int)prop.sharedMemPerBlock / 1024);
		printf("一個線程束包含的線程數量：%d\n", prop.warpSize);
		printf("一個線程塊中可包含的最大線程數量：%d\n", prop.maxThreadsPerBlock);
		printf("多維線程塊數組中每一維可包含的最大線程數量：(%d,%d,%d)\n", prop.maxThreadsDim[0],
			prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
		printf("一個線程格中每一維可包含的最大線程塊數量：(%d,%d,%d)\n", prop.maxGridSize[0],
			prop.maxGridSize[1], prop.maxGridSize[2]);
	} printf("---------------------------------------------\n");
}


void printGPU(CudaData<int>& u1, string name="") {
	vector<int> out(u1.size());
	u1.memcpyOut(out.data(), out.size());
	cout << name << "[" << u1 << "]::";
	for(size_t i = 0; i < out.size(); i++) {
		cout << out[i] << ", ";
	} cout << endl;
}
void test() {
	cout << "cuda test." << endl;
	vector<int> arr{1, 2, 3, 4, 5};

	cout << "====================== start ======================" << endl;

	/*CudaData<int> u1(1), u2;
	u1.resize(5);
	u1.memcpyIn(arr.data(), arr.size());
	
	u2=u1;

	CudaData<int> u3=u1;
	cout << u2 << ", " << u1 << ", " << u3 << endl;


	cout << "=== move ===" << endl;
	CudaData<int> uu1;
	uu1 = std::move(u1);


	CudaData<int> uu2(std::move(u2));

	// 驗證資料
	printGPU(u1, "u1 ");
	printGPU(u2, "u2 ");
	printGPU(u3, "u3 ");

	printGPU(uu1, "uu1");
	printGPU(uu2, "uu2");*/


	/*CudaData<int> b;
	b = uarr;*/

	ImgData g1("img/test.bmp");
	cuImgData ug1(g1);
	ug1.info_print(1);

	/*cout << "======= ug2 =======" << endl;
	cuImgData ug2;
	ug2=std::move(ug1);
	ug1.info_print(1);
	ug2.info_print(1);*/

	/*cout << "======= ug4 =======" << endl;
	cuImgData ug4 = std::move(ug1);
	ug1.info_print(1);
	ug4.info_print(1);*/


	/*cout << "======= ug3 =======" << endl;
	cuImgData ug3;
	ug3=ug1;
	ug3.info_print(1);*/

	/*cout << "======= ug5 =======" << endl;
	cuImgData ug5=ug1;
	ug5.info_print(1);*/


	/*ImgData out;
	cuImgData uout;
	uout.info_print();

	ImgData t1("img/test.bmp");
	cuImgData ut1(t1);
	ut1.info_print();

	ut1.out(out);

	out.info_print();
	out.bmp("__bugTest0.bmp");*/
	cout << "====================== end ======================" << endl;

}
int main(){
	//查看显卡配置
	//cuda_info();
	
	// 縮放測試
	//bilinear_test();

	// 測試代碼
	//test();
	
	// 金字塔混和
	LapBlend_Tester();

	return 0;
}