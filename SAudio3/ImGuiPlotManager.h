#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "cudaCalc.cuh"

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
namespace imGuiPlotManagerNS
{
	const int dataLoadInOneFrame = 1000;	// 1フレームで読み込めるデータ量
	const int compressSize = 10240;			// 圧縮後のデータ量(プロットできるデータ量)	long_max > shrt_max*compressSize
}

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct Compress_Data
{
	int startTime;
	int endTime;
	int usedTime;
	int compressBlock;									// dataSize/imGuiPlotManagerNS::compressSize
	long compressingData;								// 処理中のデータ
	long readPos;										// 処理位置
	int dataPos;										// 圧縮データの処理位置
	float data[imGuiPlotManagerNS::compressSize];		// 圧縮データ
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class ImGuiPlotManager
{
public:
	ImGuiPlotManager();
	~ImGuiPlotManager();

	// 圧縮波形のプロット処理
	void PlotCompressWave(std::string soundName, SoundResource *soundResource);
	
	// データカウンターリセット
	void ResetDataCnt(void) { dataCnt = imGuiPlotManagerNS::dataLoadInOneFrame; }

private:
	CUDA_CALC *cudaCalc;
	int dataCnt;										// 残り処理できるデータ量
	std::map<std::string, Compress_Data> compressData;	// [連想配列]圧縮波形のデータ

	// 圧縮データの準備
	bool InitCompressData(std::string soundName, SoundResource *soundResource);

	// 波形の圧縮処理
	void CreateCompressWave(std::string soundName, SoundResource *soundResource);
};