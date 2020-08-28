#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#pragma comment(lib, "cudart_static.lib")

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
namespace CUDACalcNS
{
	// GTX 970
	// Maximum number of threads per block:            1024
	// Max dimension size of a thread block(x, y, z) : (1024, 1024, 64)
	// Max dimension size of a grid size(x, y, z) : (2147483647, 65535, 65535)
	const int gridMaxX		= INT_MAX;
	const int blocksizeX	= 1024;	
	const int compressSize	= 1024;			// 圧縮後のデータ量(プロットできるデータ量)	long_max > shrt_max*compressSize
}

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct Compress_Data
{
	int startTime;
	//int endTime;
	int usedTime;
	int compressBlock;									// dataSize/imGuiPlotManagerNS::compressSize
	//long compressingData;								// 処理中のデータ
	//long readPos;										// 処理位置
	//int dataPos;										// 圧縮データの処理位置
	float data[CUDACalcNS::compressSize];				// 圧縮データ
};

//===================================================================================================================================
// プロトタイプ宣言
//===================================================================================================================================
class CUDA_CALC
{
public:
	CUDA_CALC() {};
	~CUDA_CALC() {};

	// カーネル CPU<->GPU
	void Kernel(short *_data, long _size, Compress_Data *_compressData);

	// テスト用プロット
	//void tmpPlot(void);
};