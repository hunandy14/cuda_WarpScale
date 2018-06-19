/*****************************************************************
Name : OpenBMP
Date : 2017/06/12
By   : CharlotteHonG
Final: 2018/06/01
*****************************************************************/
#pragma once
#include <fstream>
#include <vector>
#include <string>


//----------------------------------------------------------------
// 檔案檔頭 (BITMAPFILEHEADER)
#pragma pack(2) // 調整對齊
struct BmpFileHeader{
	uint16_t bfTybe=0x4d42;
	uint32_t bfSize;
	uint16_t bfReserved1=0;
	uint16_t bfReserved2=0;
	uint32_t bfOffBits=54;
	// constructor
	BmpFileHeader() = default;
	BmpFileHeader(uint32_t width, uint32_t height, uint16_t bits):
		bfSize(bfOffBits + width*height*bits/8)
	{
		if(bits==8) {bfSize += 1024, bfOffBits += 1024;}
	}
	// fstream
	friend std::ofstream& operator<<(std::ofstream& os, const BmpFileHeader& obj);
	friend std::ifstream& operator>>(std::ifstream& is, BmpFileHeader& obj);
	// ostream
	friend std::ostream& operator<<(std::ostream& os, const BmpFileHeader& obj);
};
#pragma pack() // 恢復對齊為預設
inline std::ofstream& operator<<(std::ofstream& os, const BmpFileHeader& obj){
	os.write((char*)&obj, sizeof(obj));
	return os;
}
inline std::ifstream& operator>>(std::ifstream& is, BmpFileHeader& obj){
	is.read((char*)&obj, sizeof(obj));
	return is;
}
inline std::ostream& operator<<(std::ostream& os, const BmpFileHeader& obj){
	using std::cout;
	using std::endl;
	cout << "# BmpFileHeader" << endl;
	cout << "    bfTybe      = " << obj.bfTybe << endl;
	cout << "    bfSize      = " << obj.bfSize << endl;
	cout << "    bfReserved1 = " << obj.bfReserved1 << endl;
	cout << "    bfReserved2 = " << obj.bfReserved2 << endl;
	cout << "    bfOffBits   = " << obj.bfOffBits;
	return os;
}


// 圖片資訊 (BITMAPINFOHEADER)
#pragma pack(2) // 調整對齊
struct BmpInfoHeader{
	uint32_t biSize=40;
	uint32_t biWidth;
	uint32_t biHeight;
	uint16_t biPlanes=1; // 1=defeaul, 0=custom
	uint16_t biBitCount;
	uint32_t biCompression=0;
	uint32_t biSizeImage;
	uint32_t biXPelsPerMeter=0; // 72dpi=2835, 96dpi=3780
	uint32_t biYPelsPerMeter=0; // 120dpi=4724, 300dpi=11811
	uint32_t biClrUsed=0;
	uint32_t biClrImportant=0;
	// constructor
	BmpInfoHeader() = default;
	BmpInfoHeader(uint32_t width, uint32_t height, uint16_t bits):
		biWidth(width), biHeight(height), biBitCount(bits),
		biSizeImage(width*height * bits/8)
	{
		if(bits==8) {biClrUsed=256;}
	}
	// fstream
	friend std::ofstream& operator<<(std::ofstream& os, const BmpInfoHeader& obj);
	friend std::ifstream& operator>>(std::ifstream& is, BmpInfoHeader& obj);
	// ostream
	friend std::ostream& operator<<(std::ostream& os, const BmpInfoHeader& obj);
};
#pragma pack() // 恢復對齊為預設
inline std::ofstream& operator<<(std::ofstream& os, const BmpInfoHeader& obj){
	os.write((char*)&obj, sizeof(obj));
	return os;
}
inline std::ifstream& operator>>(std::ifstream& is, BmpInfoHeader& obj){
	is.read((char*)&obj, sizeof(obj));
	return is;
}
inline std::ostream& operator<<(std::ostream& os, const BmpInfoHeader& obj){
	using std::cout;
	using std::endl;
	cout << "# BmpInfoHeader" << endl;
	cout << "    biSize          = " << obj.biSize << endl;
	cout << "    biWidth         = " << obj.biWidth << endl;
	cout << "    biHeight        = " << obj.biHeight << endl;
	cout << "    biPlanes        = " << obj.biPlanes << endl;
	cout << "    biBitCount      = " << obj.biBitCount << endl;
	cout << "    biCompression   = " << obj.biCompression << endl;
	cout << "    biSizeImage     = " << obj.biSizeImage << endl;
	cout << "    biXPelsPerMeter = " << obj.biXPelsPerMeter << endl;
	cout << "    biYPelsPerMeter = " << obj.biYPelsPerMeter << endl;
	cout << "    biClrUsed       = " << obj.biClrUsed << endl;
	cout << "    biClrImportant  = " << obj.biClrImportant;
	return os;
}



//----------------------------------------------------------------
class OpenBMP {
private:
	using uch = unsigned char;
public:
	// RGB 轉灰階公式
	static uch rgb2gray(const uch* p) {
		return ((
			19595 * (*(p+0))+
			38469 * (*(p+1))+
			7472  * (*(p+2))) >> 16);
	}
	// 轉灰階
	static void raw2gray(std::vector<uch>& dst, const std::vector<uch>& src) {
		if (&dst == &src) {
			// 同一個來源轉換完再resize
			for (int i = 0; i < src.size()/3; ++i)
				dst[i] = rgb2gray(&src[i*3]);
			dst.resize(src.size()/3);
		} else {
			// 通常情況先設置好大小才轉換
			dst.resize(src.size()/3);
			for (int i = 0; i < src.size()/3; ++i)
				dst[i] = rgb2gray(&src[i*3]);
		}
	}
public:
	// 讀 Bmp 檔案
	static void bmpRead(std::vector<uch>& dst, std::string name,
		uint32_t* width=nullptr, uint32_t* height=nullptr, uint16_t* bits=nullptr);
	// 寫 Bmp 檔
	static void bmpWrite(std::string name, const std::vector<uch>& src,
		uint32_t width, uint32_t height, uint16_t bits=24);
	// 讀 Raw 檔
	static void rawRead(std::vector<uch>& dst, std::string name);
	// 寫 Raw 檔
	static void rawWrite(std::string name, const std::vector<uch>& src);
};

// 讀 Bmp 檔案
inline void OpenBMP::bmpRead(std::vector<uch>& dst, std::string name,
	uint32_t* width, uint32_t* height, uint16_t* bits) {
	std::ifstream bmp(name.c_str(), std::ios::binary);
	bmp.exceptions(std::ifstream::failbit|std::ifstream::badbit);
	bmp.seekg(0, std::ios::beg);
	// 讀檔頭
	BmpFileHeader file_h;
	bmp >> file_h;
	BmpInfoHeader info_h;
	bmp >> info_h;
	// 回傳資訊
	if (width  != nullptr && 
		height != nullptr && 
		bits   != nullptr)
	{
		*width  = info_h.biWidth;
		*height = info_h.biHeight;
		*bits   = info_h.biBitCount;
	}
	// 讀 Raw
	bmp.seekg(file_h.bfOffBits, std::ios::beg);
	dst.resize(info_h.biWidth * info_h.biHeight * (info_h.biBitCount/8));
	size_t realW = info_h.biWidth * info_h.biBitCount/8.0;
	size_t alig = (realW*3) % 4;
	char* p = reinterpret_cast<char*>(dst.data());
	for(int j = info_h.biHeight-1; j >= 0; --j) {
		for(unsigned i = 0; i < info_h.biWidth; ++i) {
			// 來源是 rgb
			if(info_h.biBitCount == 24) {
				bmp.read(p + j*info_h.biWidth*3+i*3 + 2, 1);
				bmp.read(p + j*info_h.biWidth*3+i*3 + 1, 1);
				bmp.read(p + j*info_h.biWidth*3+i*3 + 0, 1);
			}
			// 來源是 gray
			else if(info_h.biBitCount == 8) {
				bmp.read(p + j*info_h.biWidth+i, 1);
			}
		}
		bmp.seekg(alig, std::ios::cur); // 跳開 4bite 對齊的空格
	}
}
// 寫 Bmp 檔
inline void OpenBMP::bmpWrite( std::string name, const std::vector<uch>& src,
	uint32_t width, uint32_t height, uint16_t bits)
{
	// 檔案資訊
	BmpFileHeader file_h(width, height, bits);
	// 圖片資訊
	BmpInfoHeader info_h(width, height, bits);
	// 寫檔
	std::ofstream bmp(name, std::ios::binary);
	bmp.exceptions(std::ifstream::failbit|std::ifstream::badbit);
	bmp << file_h << info_h;
	// 寫調色盤
	if(bits == 8) {
		for(unsigned i = 0; i < 256; ++i) {
			bmp << uch(i) << uch(i) << uch(i) << uch(0);
		}
	}
	// 寫入圖片資訊
	size_t realW = info_h.biWidth * info_h.biBitCount/8.0;
	size_t alig = (realW*3) % 4;


	for(int j = height-1; j >= 0; --j) {
		for(unsigned i = 0; i < width; ++i) {
			if(bits==24) {
				bmp << src[(j*width+i)*3 + 2];
				bmp << src[(j*width+i)*3 + 1];
				bmp << src[(j*width+i)*3 + 0];
			} else if(bits==8) {
				bmp << src[(j*width+i)];
			}
		}
		// 對齊4byte
		for(unsigned i = 0; i < alig; ++i) {
			bmp << uch(0);
		}
	}
}
// 讀 Raw 檔
inline void OpenBMP::rawRead(std::vector<uch>& dst, std::string name) {
	std::ifstream raw_file(name.c_str(), 
		std::ios::binary | std::ios::ate);
	raw_file.exceptions(std::ifstream::failbit|std::ifstream::badbit);
	dst.resize(static_cast<size_t>(raw_file.tellg()));
	raw_file.seekg(0, std::ios::beg);
	raw_file.read(reinterpret_cast<char*>(dst.data()), dst.size());
	raw_file.close();
}
// 寫 Raw 檔
inline void OpenBMP::rawWrite(std::string name, const std::vector<uch>& src) {
	std::ofstream raw_file(name.c_str(), std::ios::binary);
	raw_file.exceptions(std::ifstream::failbit|std::ifstream::badbit);
	raw_file.write(reinterpret_cast<const char*>(src.data()), src.size());
}





//----------------------------------------------------------------
struct basic_ImgData {
	std::vector<unsigned char> raw_img;
	uint32_t width;
	uint32_t height;
	uint16_t bits;
};
inline void ImgData_resize(basic_ImgData &dst, int newW, int newH, int bits) {
	dst.raw_img.resize(newW*newH*3);
	dst.width = newW;
	dst.height = newH;
	dst.bits = bits;
};
inline void ImgData_resize(const basic_ImgData& src, basic_ImgData &dst) {
	dst.raw_img.resize(src.width*src.height*3);
	dst.width = src.width;
	dst.height = src.height;
	dst.bits = src.bits;
};
inline void ImgData_write(const basic_ImgData &src, std::string name) {
	OpenBMP::bmpWrite(name, src.raw_img, src.width, src.height);
};
inline void ImgData_read(basic_ImgData &dst, std::string name) {
	OpenBMP::bmpRead(dst.raw_img, name, &dst.width, &dst.height, &dst.bits);
}



class ImgData: public basic_ImgData {
private: // 型態宣告
	using uch = unsigned char;

public: // 建構子
	ImgData() = default;
	ImgData(std::string name) {
		OpenBMP::bmpRead(raw_img, name, &width, &height, &bits);
	}
	ImgData(std::vector<uch>& raw_img, uint32_t width, uint32_t height, uint16_t bits):
		basic_ImgData({raw_img, width, height, bits}) {}
	ImgData(uint32_t width, uint32_t height, uint16_t bits) {
		raw_img.resize(width*height* bits/8);
		this->width  = width;
		this->height = height;
		this->bits   = bits;
	}
	explicit ImgData(basic_ImgData& imgData): basic_ImgData(imgData) {}

public: // 存取方法
	inline uch& operator[](size_t idx) {
		return this->raw_img[idx];
	}
	inline const uch& operator[](size_t idx) const {
		return this->raw_img[idx];
	}
	inline uch* at2d(size_t y, size_t x) {
		return &raw_img[(y*width + x) *(bits>>3)];
	}
	inline const uch* at2d(size_t y, size_t x) const {
		return &raw_img[(y*width + x) *(bits>>3)];
	}
	// 線性插值(快速測試用, RGB效率很差)
	std::vector<double> at2d_linear(double y, double x) const { 
		std::vector<double> RGB(bits>>3);
		// 整數就不算了
		if (y==(int)y && x==(int)x) {
			auto p = this->at2d(y, x);
			for (int i = 0; i < RGB.size(); i++)
				RGB[i] = static_cast<double>(p[i]);
			return RGB;
		}
		// 獲取鄰點
		int x0 = (int)(x);
		int x1 = (x)==(int)(x)? (int)(x): (int)(x+1.0);
		int y0 = (int)(y);
		int y1 = (y)==(int)(y)? (int)(y): (int)(y+1.0);
		// 獲取比例
		double dx1 = x -  x0;
		double dx2 = 1 - dx1;
		double dy1 = y -  y0;
		double dy2 = 1 - dy1;
		// 計算插值
		for (int i = 0; i < RGB.size(); i++) {
			// 獲取點
			const double& A = raw_img[(y0*width + x0)*(bits>>3) + i];
			const double& B = raw_img[(y0*width + x1)*(bits>>3) + i];
			const double& C = raw_img[(y1*width + x0)*(bits>>3) + i];
			const double& D = raw_img[(y1*width + x1)*(bits>>3) + i];
			// 乘出比例(要交叉)
			double AB = A*dx2 + B*dx1;
			double CD = C*dx2 + D*dx1;
			RGB[i] = AB*dy2 + CD*dy1;
		}
		return RGB;
	}

public: // 大小方法
	friend inline bool operator!=(const ImgData& lhs, const ImgData& rhs) {
		return !(lhs == rhs);
	}
	friend inline bool operator==(const ImgData& lhs, const ImgData& rhs) {
		if (lhs.width == rhs.width &&  lhs.height == rhs.height) {
			return 1;
		} return 0;
	}
	std::size_t size() const {
		return this->raw_img.size();
	}
	void resize(uint32_t width, uint32_t height, uint16_t bits) {
		raw_img.resize(width*height * bits>>3);
		this->width  = width;
		this->height = height;
		this->bits   = bits;
	}
	void resize(const ImgData& src) {
		resize(src.width, src.height, src.bits);
	}
	void clear() {
		raw_img.clear();
		this->width  = 0;
		this->height = 0;
		this->bits   = 0;
	}
	void info_print() const {
		std::cout << ">>IMG basic info:" << "\n";
		std::cout << "  - img size  = " << this->size() << "\n";
		std::cout << "  - img width = " << this->width  << "\n";
		std::cout << "  - img heigh = " << this->height << "\n";
		std::cout << "  - img bits  = " << this->bits   << "\n\n";
	}

public: // 自訂方法
	void bmp(std::string name) const {
		OpenBMP::bmpWrite(name, raw_img, width, height, bits);
	}
	ImgData& convertGray() {
		if (bits == 24) {
			OpenBMP::raw2gray(raw_img, raw_img);
			bits = 8;
		}
		return *this;
	}
	ImgData toConvertGray() const {
		ImgData img;
		img.resize(*this);
		if (bits == 24) {
			OpenBMP::raw2gray(img.raw_img, raw_img);
			img.bits = 8;
		} else {
			img = *this;
		}
		return img;
	}
	ImgData toSnip (uint32_t width, uint32_t height, uint32_t y=0, uint32_t x=0) const {
		// 檢查超過邊界
		if (width+x > this->width || height+y > this->height)
			throw std::out_of_range("toSnip() out of range");
		// 開始擷取
		ImgData img(width, height, this->bits);
		for (int j = 0; j < img.height; j++) {
			for (int i = 0; i < img.width; i++) {
				auto srcIt = this->at2d(j+y, i+x);
				auto dstIt = img.at2d(j, i);
				for (size_t rgb = 0; rgb < bits>>3; rgb++) {
					dstIt[rgb] = srcIt[rgb];
				}
			}
		}
		return img;
	}

public: // 畫線

};


class ImgData_nor: public basic_ImgData {
public:
	ImgData_nor(std::string name) {
		OpenBMP::bmpRead(raw_img, name, &width, &height, &bits);
		Normalization();
		raw_img.clear();
	}
	void write(std::string name) {
		reNormalization();
		OpenBMP::bmpWrite(name, raw_img, width, height, bits);
		nor_img.clear();
	}
public:
	void Normalization() {
		nor_img.resize(raw_img.size());
		#pragma omp parallel for
		for (int i = 0; i < nor_img.size(); i++) {
			nor_img[i] = raw_img[i] /255.0;
		}
	}
	void reNormalization() {
		raw_img.resize(nor_img.size());
		#pragma omp parallel for
		for (int i = 0; i < nor_img.size(); i++) {
			raw_img[i] = nor_img[i] *255.0;
		}
	}
public:
	std::vector<float> nor_img;
};

