/*****************************************************************
Name :
Date : 2018/04/12
By   : CharlotteHonG
Final: 2018/04/12
*****************************************************************/
#pragma once
#include "OpenBMP.hpp"

// 混合原始圖
void LapBlender(basic_ImgData & dst, 
	const basic_ImgData & src1, 
	const basic_ImgData & src2, 
	double ft, int mx, int my
);

// 範例測試
void LapBlend_Tester();
