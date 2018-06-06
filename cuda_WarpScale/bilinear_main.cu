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

#include "bilinear.cuh"
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
	// Ū��
	ImgData src("img/test.bmp");
	//ImgData src("737400.bmp");
	ImgData srcGray, dst, temp;
	srcGray = src.toConvertGray();

	// GPU�t��
	double ratio = 5;
	vector<float> img_gpuRst, img_data = tofloat(srcGray.raw_img.data(), srcGray.size());
	
	T.start();
	cuWarpScale_rgb(src, dst, ratio);
	T.print(" cuWarpScale_rgb");
	dst.bmp("cutestImg.bmp");

	// CPU�t��
	T.start();
	WarpScale_rgb(src, dst, ratio);
	T.print(" WarpScale_rgb");
	dst.bmp("testImg.bmp");

	return 0;
}