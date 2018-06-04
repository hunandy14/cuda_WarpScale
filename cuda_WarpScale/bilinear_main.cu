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
#include "CudaMem\CudaMem.cuh"
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
	// 讀取
	ImgData img1("img//kanna.bmp");
	T.start();
	img1.convertGray();
	T.print("轉灰階圖");

	// 處理
	float ratio = 5;
	vector<float> img_gpuRst, img_data = tofloat(img1.raw_img.data(), img1.size());

	double time;
	//time = biliner_texture(img_gpuRst, img_data, img1.width, img1.height, ratio);
	time = biliner_share(img_gpuRst, img_data, img1.width, img1.height, ratio);
	//time = biliner_CPU(img_gpuRst, img_data, img1.width, img1.height, ratio);


	// 輸出
	vector<unsigned char> img_out =  touch(img_gpuRst.data(), img_gpuRst.size());
	//string name = "img//Out-texture_"+to_string(time)+".bmp";
	string name = "GpuOut.bmp";
	OpenBMP::bmpWrite(name.c_str(), img_out, img1.width*ratio, img1.height*ratio, 8);

	return 0;
}