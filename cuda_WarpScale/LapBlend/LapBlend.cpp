/*****************************************************************
Name :
Date : 2018/04/12
By   : CharlotteHonG
Final: 2018/04/12
*****************************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <timer.hpp>
using namespace std;

#include "LapBlend.hpp"
#include "cubilinear.hpp"

#define or ||
#define and &&
#define LAP_OCTVS 5

//==================================================================================
// 轉換
//==================================================================================
// 重設 ImgData 大小
void ImgData_resize(basic_ImgData &dst, int newW, int newH, int bits) {
	dst.raw_img.resize(newW*newH*3);
	dst.width = newW;
	dst.height = newH;
	dst.bits = bits;
};
void ImgData_resize(const basic_ImgData& src, basic_ImgData &dst) {
	dst.raw_img.resize(src.width*src.height*3);
	dst.width = src.width;
	dst.height = src.height;
	dst.bits = src.bits;
};
// 輸出 bmp
void ImgData_write(const basic_ImgData &src, string name) {
	OpenBMP::bmpWrite(name, src.raw_img, src.width, src.height);
};
// 讀取bmp
void ImgData_read(basic_ImgData &dst, std::string name) {
	OpenBMP::bmpRead(dst.raw_img, name, &dst.width, &dst.height, &dst.bits);
}



//==================================================================================
// 圖片放大縮小
//==================================================================================
// 快速 線性插值
inline static void fast_Bilinear_rgb(unsigned char* p, 
	const basic_ImgData& src, double y, double x)
{
	// 起點
	int _x = (int)x;
	int _y = (int)y;
	// 左邊比值
	double l_x = x - (double)_x;
	double r_x = 1.f - l_x;
	double t_y = y - (double)_y;
	double b_y = 1.f - t_y;
	int srcW = src.width;
	int srcH = src.height;

	// 計算RGB
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

	*(p+0) = (unsigned char) R;
	*(p+1) = (unsigned char) G;
	*(p+2) = (unsigned char) B;
}
// 快速補值
inline static void fast_NearestNeighbor_rgb(unsigned char* p,
	const basic_ImgData& src, double y, double x) 
{
	// 位置(四捨五入)
	int _x = (int)(x+0.5);
	int _y = (int)(y+0.5);
	int srcW = src.width;
	int srcH = src.height;

	// 計算RGB
	double R , G, B;
	int x2 = (_x+1) > src.width -1? src.width -1: _x+1;
	int y2 = (_y+1) > src.height-1? src.height-1: _y+1;
	R  = (double)src.raw_img[(y2 * srcW + x2) *3 + 0];
	G  = (double)src.raw_img[(y2 * srcW + x2) *3 + 1];
	B  = (double)src.raw_img[(y2 * srcW + x2) *3 + 2];

	*(p+0) = (unsigned char) R;
	*(p+1) = (unsigned char) G;
	*(p+2) = (unsigned char) B;
}


//==================================================================================
// 金字塔處理
//==================================================================================
// 金字塔
using cuLapPyr = vector<cuImgData>;
void buildLaplacianPyramids(const cuImgData &usrc, cuLapPyr &upyr, int octvs=LAP_OCTVS) {
	upyr.resize(octvs);
	//upyr[0] = std::move(usrc);// todo 警告 這裡的移動語意還沒做(有做防double delete)
	imgCopy(usrc, upyr[0]);

	cuImgData utemp(usrc.width, usrc.height, usrc.bits);
	cuImgData& uExpend=utemp;

	for(int i = 1; i < octvs; i++) {
		cuImgData& uReduce = upyr[i]; // this is temp
		// preImg 縮小+模糊 到 uReduce
		WarpScale_rgb(upyr[i-1], utemp, 0.5);
		GaussianBlur(utemp, uReduce, 3);
		// uReduce 放大到 uExpend, then preImg -= uExpend
		WarpScale_rgb(uReduce, uExpend, 2.0);
		imgSub(upyr[i-1], uExpend);
	}
}
// 混合拉普拉斯金字塔
void blendLaplacianPyramids(cuLapPyr& LS, const cuLapPyr& LA, const cuLapPyr& LB) {
	LS.resize(LA.size());

	// 混合圖片
	for(int idx = 0; idx < LS.size(); idx++) {
		// 初始化
		cuImgData& dst = LS[idx];
		dst.resize(LA[idx].width, LA[idx].height, LA[idx].bits);
		// 開始混合各層
		if(idx == LS.size()-1) {
			imgBlendAlpha(LA[idx], LB[idx], dst);
		} else {
			imgBlendHalf(LA[idx], LB[idx], dst);
		}
	}
}
// 解拉普拉斯金字塔
void reLaplacianPyramids(cuLapPyr &upyr, cuImgData &udst, int octvs=LAP_OCTVS) {
	Timer t1;
	int newH = (int)(upyr[0].height);
	int newW = (int)(upyr[0].width);
	
	for(int i = octvs-1; i >= 1; i--) {
		cuImgData expend;
		WarpScale_rgb(upyr[i], expend, 2.0);
		imgAdd(upyr[i-1], expend);
	}

	udst.resize(newW, newH, upyr[0].bits);
	//udst = std::move(upyr[0]); // todo 警告 這裡的 move 還沒做
	imgCopy(upyr[0], udst);
}
// 混合圖片
void blendLaplacianImg(cuImgData& udst, const cuImgData& usrc1, const cuImgData& usrc2) {
	Timer t1;
	t1.priSta=0;

	cuLapPyr uLA, uLB, uLS;// todo 警告 move 還沒做 uLS 解構之後傳出去的 udst 就毀了

	// 拉普拉斯金字塔 AB
	t1.start();
	buildLaplacianPyramids(usrc1, uLA);
	t1.print("    buildLapA");
	t1.start();
	buildLaplacianPyramids(usrc2, uLB);
	t1.print("    buildLapB");

	// 混合金字塔
	t1.start();
	blendLaplacianPyramids(uLS, uLA, uLB);
	t1.print("    blendImg");

	// 還原拉普拉斯金字塔
	t1.start();
	reLaplacianPyramids(uLS, udst);
	t1.print("    rebuildLaplacianPyramids");
}


//==================================================================================
// 圓柱投影
//==================================================================================
// 圓柱投影座標反轉換
inline static
void WarpCylindrical_CoorTranfer_Inve(double R,
	size_t width, size_t height, double& x, double& y)
{
	double r2 = (x - width*.5);
	double k = sqrt(R*R + r2*r2) / R;
	x = (x - width *.5)*k + width *.5;
	y = (y - height*.5)*k + height*.5;
}
// 圓柱投影 basic_ImgData
void WarpCylindrical(basic_ImgData &dst, const basic_ImgData &src, 
	double R ,int mx=0, int my=0, double edge=0.0)
{
	int srcW = src.width;
	int srcH = src.height;
	int moveH = (srcH*edge) + my;
	int moveW = mx;

	int dstW = srcW+moveW;
	int dstH = srcH * (1+edge*2);
	ImgData_resize(dst, dstW, dstH, src.bits);

	// 圓柱投影
#pragma omp parallel for
	for (int j = 0; j < srcH; j++){
		for (int i = 0; i < srcW; i++){
			double x = i, y = j;
			WarpCylindrical_CoorTranfer_Inve(R, srcW, srcH, x, y);
			if (x >= 0 && y >= 0 && x < srcW - 1 && y < srcH - 1) {
				unsigned char* p = &dst.raw_img[((j+moveH)*(srcW+moveW) + (i+moveW)) *3];
				fast_Bilinear_rgb(p, src, y, x);
			}
		}
	}
}
// 找到圓柱投影角點
void WarpCyliCorner(const basic_ImgData &src, vector<int>& corner) {
	corner.resize(6);
	// 左上角角點
	for (int i = 0; i < src.width; i++) {
		int pix = (int)src.raw_img[(src.height/2*src.width +i)*3 +0];
		if (i<src.width/2 and pix != 0) {
			corner[0]=i;
			//cout << "corner=" << corner[0] << endl;
			i=src.width/2;
		} else if (i>src.width/2 and pix == 0) {
			corner[2] = i-1;
			//cout << "corner=" << corner[2] << endl;
			break;
		}
	}
	// 右上角角點
	for (int i = 0; i < src.height; i++) {
		int pix = (int)src.raw_img[(i*src.width +corner[0])*3 +0];
		if (i<src.height/2 and pix != 0) {
			corner[1] = i;
			//cout << "corner=" << corner[2] << endl;
			i=src.height/2;
		} else if (i>src.height/2 and pix == 0) {
			corner[3] = i-1;
			//cout << "corner=" << corner[3] << endl;
			break;
		}
	}
}
// 刪除左右黑邊
void delPillarboxing(const basic_ImgData &src, basic_ImgData &dst,
	vector<int>& corner)
{
	// 新圖大小
	int newH=src.height;
	int newW=corner[2]-corner[0];
	ImgData_resize(dst, newW, newH, 24);
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW; i++) {
			for (int  rgb = 0; rgb < 3; rgb++) {
				dst.raw_img[(j*dst.width+i)*3 +rgb] =
					src.raw_img[(j*src.width+(i+corner[0]))*3 +rgb];
			}
		}
	}
	ImgData_write(dst, "delPillarboxing.bmp");
}
// 取出重疊區
void getOverlap(const basic_ImgData &src1, const basic_ImgData &src2,
	basic_ImgData& cut1, basic_ImgData& cut2, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=corner[3]-corner[1]-abs(my);
	int newW=corner[2]-corner[0]+mx;
	// 重疊區大小
	int lapH=newH;
	int lapW=corner[2]-corner[0]-mx;
	// 兩張圖的高度偏差值
	int myA = my<0? 0:my;
	int myB = my>0? 0:-my;
	// 重疊區
	ImgData_resize(cut1, lapW, lapH, 24);
	ImgData_resize(cut2, lapW, lapH, 24);
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW-mx; i++) {
			// 圖1
			if (i < corner[2]-corner[0]-mx) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut1.raw_img[(j*cut1.width +i) *3+rgb] = 
						src1.raw_img[(((j+myA)+corner[1])*src1.width +(i+corner[0]+mx)) *3+rgb];
				}
			}
			// 圖2
			if (i >= mx) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut2.raw_img[(j*cut2.width +(i-mx)) *3+rgb] = 
						src2.raw_img[(((j+myB)+corner[1])*src2.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
	//ImgData_write(cut1, "__cut1.bmp");
	//ImgData_write(cut2, "__cut2.bmp");
}
// 取出重疊區(沒有裁減)
void getOverlap_noncut(const basic_ImgData &src1, const basic_ImgData &src2,
	basic_ImgData& cut1, basic_ImgData& cut2, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=src1.height+abs(my);
	int newW=corner[2]-corner[0]+mx;
	// 重疊區大小
	int lapH=newH;
	int lapW=corner[2]-corner[0]-mx;
	// 兩張圖的高度偏差值
	int myA = my>0? 0:-my;
	int myB = my<0? 0:my;
	// 重疊區
	ImgData_resize(cut1, lapW, lapH, 24);
	ImgData_resize(cut2, lapW, lapH, 24);
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW-mx; i++) {
			// 圖1
			if (i < corner[2]-corner[0]-mx and j<src1.height-1) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut1.raw_img[((j+myA)*cut1.width +i) *3+rgb]
						= src1.raw_img[((j)*src1.width +(i+corner[0]+mx)) *3+rgb];
				}
			}
			// 圖2
			if (i >= mx and j<src2.height-1) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut2.raw_img[((j+myB)*cut2.width +(i-mx)) *3+rgb] = 
						src2.raw_img[((j)*src1.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
	//ImgData_write(cut1, "__cut1.bmp");
	//ImgData_write(cut2, "__cut2.bmp");
}
// 重疊區與兩張原圖合併
void mergeOverlap(const basic_ImgData &src1, const basic_ImgData &src2,
	const basic_ImgData &blend, basic_ImgData &dst, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=corner[3]-corner[1]-abs(my);
	int newW=corner[2]-corner[0]+mx;
	ImgData_resize(dst, newW, newH, 24);
	// 兩張圖的高度偏差值
	int myA = my<0? 0:my;
	int myB = my>0? 0:-my;

	// 合併圖片
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW; i++) {
			// 圖1
			if (i < mx) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = src1.raw_img[(((j+myA)+corner[1])*src1.width +(i+corner[0])) *3+rgb];
				}
			}
			// 重疊區
			else if (i >= mx and i < corner[2]-corner[0]) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = blend.raw_img[(j*blend.width+(i-mx)) *3+rgb];
				}
			}
			// 圖2
			else if (i >= corner[2]-corner[0]) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = src2.raw_img[(((j+myB)+corner[1])*src1.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
}

void mergeOverlap_noncut(const basic_ImgData &src1, const basic_ImgData &src2,
	const basic_ImgData &blend, basic_ImgData &dst, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=src1.height+abs(my);
	int newW=corner[2]-corner[0]+mx;
	ImgData_resize(dst, newW, newH, 24);
	// 兩張圖的高度偏差值
	int myA = my>0? 0:-my;
	int myB = my<0? 0:my;

	// 合併圖片
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW; i++) {
			// 圖1
			if (i < mx and j<src1.height-1) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[((j+myA)*dst.width +i) *3+rgb] = 
						src1.raw_img[(((j))*src1.width +(i+corner[0])) *3+rgb];
				}
			}
			// 重疊區
			else if (i >= mx and i < corner[2]-corner[0]) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = 
						blend.raw_img[(j*blend.width+(i-mx)) *3+rgb];
				}
			}
			// 圖2
			else if (i >= corner[2]-corner[0] and j<src2.height) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[((j+myB)*dst.width +i) *3+rgb] = 
						src2.raw_img[((j)*src1.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
}


//==================================================================================
// 混合兩張投影過(未裁減)的圓柱，過程會自動裁減輸出
void WarpCyliMuitBlend(cuImgData &udst, 
	const cuImgData &usrc1, const cuImgData &usrc2,
	const vector<int>& corner) 
{
	Timer t1;
	t1.priSta=1;

	// 暫存
	cuImgData ucut1, ucut2, ublend;

	// 取出重疊區
	t1.start();
	getOverlap(usrc1, usrc2, ucut1, ucut2, corner); // 5ms
	t1.print("  getOverlap");

	// 混合重疊區
	t1.start();
	blendLaplacianImg(ublend, ucut1, ucut2); // 53ms -> 21ms->19ms
	t1.print("  blendLaplacianImg");
	
	// 合併三張圖片
	t1.start();
	mergeOverlap(usrc1, usrc2, ublend, udst, corner); // 5ms -> 2ms
	t1.print("  mergeOverlap");
}


//==================================================================================
// 公開函式
//==================================================================================
// 混合原始圖
void LapBlender(basic_ImgData &dst, 
	const basic_ImgData &src1, const basic_ImgData &src2,
	double ft, int mx, int my)
{
	Timer t;
	basic_ImgData warp1, warp2;
	t.start();
	WarpCylindrical(warp1, src1, ft);
	WarpCylindrical(warp2, src2, ft);
	t.print("WarpCylindrical"); // 20ms
	
	// 檢測圓柱圖角點(minX, minY, maxX, maxY, mx, my)
	vector<int> corner{0, 0, 0, 0, mx, my};
	WarpCyliCorner(warp1, corner); // 0ms

	// 混合圖像
	cuImgData uwarp1(warp1), uwarp2(warp2), udst;
	t.start();
	WarpCyliMuitBlend(udst, uwarp1, uwarp2, corner); // 31ms
	t.print("WarpCyliMuitBlend");

	udst.out(dst);
}



// 範例程式
void LapBlend_Tester() {
	basic_ImgData src1, src2, dst;
	string name1, name2;
	double ft; int Ax, Ay;

	// 籃球 (1334x1000, 237ms)
	name1="img/ball_01.bmp", name2="img/ball_02.bmp"; ft=2252.97, Ax=539, Ay=-2;
	// 校園 (752x500, 68ms)
	//name1="img/sc02.bmp", name2="img/sc03.bmp"; ft=676.974, Ax=216, Ay=4;

	// 讀取圖片
	ImgData_read(src1, name1);
	ImgData_read(src2, name2);

	// 混合圖片
	Timer t1;
	LapBlender(dst, src1, src2, ft, Ax, Ay);
	t1.print(" LapBlender");
	// 輸出圖片
	ImgData_write(dst, "_WarpCyliMuitBlend.bmp");
}
//==================================================================================
