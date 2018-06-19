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
using namespace std;

#include "Timer.hpp"
#include "LapBlend.hpp"
#include "cubilinear.hpp"

#define LAP_OCTVS 5

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
// 找到圓柱投影角點
void WarpCyliCorner(const basic_ImgData &src, vector<int>& corner) {
	corner.resize(6);
	int srcW = src.width;
	int srcH = src.height;

	// 左上角角點
	for (int i = 0; i < srcW; i++) {
		int pix = (int)src.raw_img[(srcH/2*srcW +i)*3 +0];
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
	// 右上角角點
	for (int j = 0; j < srcH; j++) {
		int pix = (int)src.raw_img[(j*srcW +corner[0])*3 +0];
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


//==================================================================================
// 混合兩張投影過(未裁減)的圓柱，過程會自動裁減輸出
void WarpCyliMuitBlend(cuImgData &udst, 
	const cuImgData &usrc1, const cuImgData &usrc2,
	const vector<int>& corner) 
{
	Timer t1;
	t1.priSta=0;

	// 暫存
	cuImgData ucut1, ucut2, ublend;

	// 取出重疊區
	t1.start();
	getOverlap(usrc1, usrc2, ucut1, ucut2, corner); // 5ms
	t1.print("   getOverlap");

	// 混合重疊區
	t1.start();
	blendLaplacianImg(ublend, ucut1, ucut2); // 53ms -> 21ms->19ms
	t1.print("   blendLaplacianImg");
	
	// 合併三張圖片
	t1.start();
	mergeOverlap(usrc1, usrc2, ublend, udst, corner); // 5ms -> 2ms
	t1.print("   mergeOverlap");
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
	t.priSta=1;
	cuImgData uwarp1, uwarp2;
	cuImgData usrc1(src1), usrc2(src2);
	cuImgData udst;

	t.start();
	WarpCylindrical(usrc1, uwarp1, ft);
	WarpCylindrical(usrc2, uwarp2, ft);
	t.print("  WarpCylindrical"); // 20ms->2+2ms

	// 檢測圓柱圖角點(minX, minY, maxX, maxY, mx, my)
	vector<int> corner{0, 0, 0, 0, mx, my};
	CudaData<int> ucorner(6);
	t.start();
	// todo 這裡不知道怎麼解 gpu跟cpu運算都是0ms
	// 卡在warpData在gpu上 就算算完4點再拿出來也超費時
	WarpCyliCorner(uwarp1, ucorner, mx, my); 
	ucorner.memcpyOut(corner.data(), corner.size());
	//basic_ImgData warp1; uwarp1.out(warp1);
	//WarpCyliCorner(warp1, corner); // 0ms
	t.print("  WarpCyliCorner"); // 8ms->6ms

	// 混合圖像
	t.start();
	WarpCyliMuitBlend(udst, uwarp1, uwarp2, corner); // 31ms
	t.print("  WarpCyliMuitBlend");

	// 輸出影像
	t.start();
	udst.out(dst);
	t.print("  ##DATA OUT");
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
