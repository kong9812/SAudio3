#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "SoundBase.h"
#include "cudaCalc.cuh"

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
	//void ResetDataCnt(void) { dataCnt = imGuiPlotManagerNS::dataLoadInOneFrame; }

private:
	CUDA_CALC *cudaCalc;
	int dataCnt;										// 残り処理できるデータ量
	std::map<std::string, Compress_Data> compressData;	// [連想配列]圧縮波形のデータ

	// 圧縮データの準備
	//bool InitCompressData(std::string soundName, SoundResource *soundResource);

	// 波形の圧縮処理
	//void CreateCompressWave(std::string soundName, SoundResource *soundResource);
};