/*****************************************************************
Name : 
Date : 2018/01/09
By   : CharlotteHonG
Final: 2018/01/09
*****************************************************************/
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <class T>
class CudaData_type {

};




// Cuda 記憶體自動管理程序
template <class T>
class CudaData {
public:
	CudaData(){}
	CudaData(size_t size){
		malloc(size);
	}
	CudaData(const T* dataIn ,size_t size): len(size){
		malloc(size);
		memcpyIn(dataIn, size);
	}
	~CudaData(){
		if(gpuData!=nullptr) {
			cudaFree(gpuData);
			gpuData = nullptr;
			len = 0;
		}
	}
public:
	void malloc(size_t size) {
		this->~CudaData();
		len = size;
		cudaMalloc((void**)&gpuData, size*sizeof(T));
	}
	void memcpyIn(const T* dataIn ,size_t size) {
		if(size > len) {throw out_of_range("memcpyIn input size > curr size.");}
		cudaMemcpy(gpuData, dataIn, size*sizeof(T), cudaMemcpyHostToDevice);
	}
	void memcpyOut(T* dst ,size_t size) {
		cudaMemcpy(dst, gpuData, size*sizeof(T), cudaMemcpyDeviceToHost);
	}
	void memset(int value, size_t size) {
		if(size>len) {
			throw out_of_range("memset input size > curr size.");
		}
		cudaMemset(gpuData, value, size*sizeof(T));
	}
	size_t size() {
		return this->len;
	}
public:
	operator T*() {
		return gpuData;
	}
	operator const T*() const {
		return gpuData;
	}
private:
	T* gpuData = nullptr;
	size_t len = 0;
};

// 第一次malloc非常耗時
static CudaData<int> __CudaDataInit__(0);



// CudaArr 記憶體自動管理程序
template <class T>
class CudaMemArr {
public:
	CudaMemArr(){}
	CudaMemArr(size_t dstW, size_t dstH):
		width(dstW), height(dstH)
	{
		malloc(dstW, dstH);
	}
	CudaMemArr(const T* dataIn ,size_t dstW, size_t dstH): 
		width(dstW), height(dstH)
	{
		malloc(dstW, dstH);
		memcpyIn(dataIn, dstW*dstH);
	}
	~CudaMemArr(){
		if(cuArray!=nullptr) {
			cudaFreeArray(cuArray);
			cuArray = nullptr;
			width = height = 0;
		}
	}
public:
	void malloc(size_t dstW, size_t dstH) {
		this->~CudaMemArr();
		cudaMallocArray(&cuArray, &chDesc, dstW, dstH);
	}
	void memcpyIn(const T* dataIn ,size_t size) {
		if(size > width*height) {throw out_of_range("memcpyIn input size > curr size.");}
		cudaMemcpyToArray(cuArray, 0, 0, dataIn, size*sizeof(T), cudaMemcpyHostToDevice);
	}
	size_t size() {
		return width*height;
	}
public:
	operator cudaArray*() {
		return cuArray;
	}
public:
	const cudaChannelFormatDesc chDesc = cudaCreateChannelDesc<T>();
private:
	size_t width = 0;
	size_t height = 0;
	cudaArray* cuArray = nullptr;
};