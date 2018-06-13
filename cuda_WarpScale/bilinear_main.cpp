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

int main(){

	Timer T;
	double ratio = 1;
	// 讀取
	ImgData src("img/test.bmp"); ratio = 0.5;
	//ImgData src("img/737400.bmp"); ratio = 1;
	ImgData dst;
	dst.resize(src.width*ratio, src.height*ratio, src.bits);



	// GPU速度
	//T.start();
	//cuWarpScale_kernel_test(src, dst, ratio);
	WarpScale_rgb_test(src, dst, ratio);
	//T.print(" cuWarpScale_rgb");

	dst.bmp("cutestImg.bmp");

	// CPU速度
	T.start();
	WarpScale_rgb(src, dst, ratio);
	T.print(" WarpScale_rgb");
	dst.bmp("testImg.bmp");

	return 0;
}