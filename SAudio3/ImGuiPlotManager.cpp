//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "ImguiManager.h"
#include "ImGuiPlotManager.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
ImGuiPlotManager::ImGuiPlotManager()
{
	//// 初期化
	//dataCnt = imGuiPlotManagerNS::dataLoadInOneFrame;	// 残り処理できるデータ数
	cudaCalc = new CUDA_CALC;
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
ImGuiPlotManager::~ImGuiPlotManager()
{
	// 連想配列の削除
	if (conversionData.size() > NULL)
	{
		auto begin = conversionData.begin();
		auto end = conversionData.end();
		for (auto i = begin; i != end; i++)
		{
			// [チャンネルごと]データ部の削除
			for (int j = 0; j < i->second.channel; j++)
			{
				SAFE_DELETE_ARRAY(i->second.data[j])
			}
			SAFE_DELETE(i->second.data)
		}
		conversionData.clear();
	}
	SAFE_DELETE(cudaCalc)
}

//===================================================================================================================================
// 圧縮波形のプロット処理
//===================================================================================================================================
void ImGuiPlotManager::PlotCompressWave(std::string soundName, SoundResource *soundResource)
{
	if (!soundResource->isWaveUpdate && soundResource->isCompressed)
	{
		// プロット
		Conversion_Data *tmpConversionData = &conversionData[soundName];
		ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
		for (int i = 0; i < soundResource->waveFormatEx.nChannels; i++)
		{
			// Draw first plot with multiple sources
			ImGui::PlotConfig conf;
			conf.values.count = tmpConversionData->sampingPerChannel;
			conf.values.ys_list = (const float **)&tmpConversionData->data[i]; // use ys_list to draw several lines simultaneously
			conf.values.ys_count = 1;
			conf.values.colors = (const ImU32*)&ImColor(IM_COL32_BLACK);

			conf.scale.min = -1;
			conf.scale.max = 1;

			conf.tooltip.show = true;

			conf.grid_x.show = false;
			//conf.grid_x.size = CUDACalcNS::compressSize;
			//conf.grid_x.subticks = 1;
			conf.grid_y.show = false;
			//conf.grid_y.size = 1.0f;
			//conf.grid_y.subticks = 5;

			conf.selection.show = false;
			//conf.selection.start = &selection_start;
			//conf.selection.length = &selection_length;
			conf.frame_size = plotextent;
			ImGui::Plot("plot1", conf);

			//ImGui::PlotLines("", tmpConversionData->data[i], tmpConversionData->sampingPerChannel, 0, "", FLT_MAX, FLT_MAX, plotextent);
		}
		ImGui::Text("usedTime:%d", tmpConversionData->usedTime);
		ImGui::Text("size:%ld", soundResource->size);
	}
	else
	{
		// 変換処理
		conversionData[soundName] = cudaCalc->conversion(soundResource->data, soundResource->size, soundResource->waveFormatEx.nChannels);
		soundResource->isWaveUpdate = !soundResource->isWaveUpdate;
		soundResource->isCompressed = !soundResource->isCompressed;

		//// 圧縮処理
		//compressData[soundName] = cudaCalc->compressor(soundResource->data, soundResource->size, soundResource->waveFormatEx.nChannels);
		//soundResource->isWaveUpdate = !soundResource->isWaveUpdate;
		//soundResource->isCompressed = !soundResource->isCompressed;

		//Compress_Data *tmpCompressData = &compressData[soundName];
		//CreateCompressWave(soundName, soundResource);
		//ImGui::Text("compressBlock:%d", tmpCompressData->compressBlock);
		//ImGui::Text("readPos:%ld", tmpCompressData->readPos);
		//ImGui::Text("dataPos:%d", tmpCompressData->dataPos);
		//ImGui::Text("size:%ld", soundResource->size);
	}
}

////===================================================================================================================================
//// 圧縮データの準備
////===================================================================================================================================
//bool ImGuiPlotManager::InitCompressData(std::string soundName, SoundResource *soundResource)
//{
//	// 再描画の必要がある
//	if (soundResource->isWaveUpdate)
//	{
//		// 圧縮完了していない
//		if (!soundResource->isCompressed)
//		{
//			// [準備]圧縮処理(初期化されていなければここに入る)
//			Compress_Data *tmpCompressData = &compressData[soundName];
//			if (tmpCompressData->compressBlock == NULL)
//			{
//				// データのサイズが圧縮量より小さい
//				if ((soundResource->size / sizeof(short)) < imGuiPlotManagerNS::compressSize)
//				{
//					tmpCompressData->compressBlock = soundResource->size / sizeof(short);
//				}
//				else
//				{
//					tmpCompressData->compressBlock = ((soundResource->size / sizeof(short)) / imGuiPlotManagerNS::compressSize);
//				}
//				// 処理位置の初期化
//				tmpCompressData->readPos = NULL;
//				tmpCompressData->dataPos = NULL;
//				tmpCompressData->startTime = timeGetTime();
//			}
//		}
//		else
//		{
//			return false;
//		}
//	}
//	else
//	{
//		return false;
//	}
//	return true;
//}

////===================================================================================================================================
//// 波形の圧縮処理
////===================================================================================================================================
//void ImGuiPlotManager::CreateCompressWave(std::string soundName, SoundResource *soundResource)
//{
//	if (dataCnt <= NULL) { return; }	// まだお越しくださいませー(このフレームはこれ以上処理できません！)
//
//	// 圧縮データの準備(圧縮処理必要ないデータを飛ばす)
//	if (InitCompressData(soundName, soundResource))
//	{
//		Compress_Data *tmpCompressData = &compressData[soundName];
//
//		while (dataCnt > NULL)
//		{
//			// 圧縮が未完了ならここに入る
//			if (tmpCompressData->dataPos < imGuiPlotManagerNS::compressSize)
//			{
//				// 圧縮開始!
//				tmpCompressData->compressingData += soundResource->data[tmpCompressData->readPos];
//				tmpCompressData->readPos++;
//
//				// 数が足りたら
//				if ((tmpCompressData->readPos % tmpCompressData->compressBlock) == NULL)
//				{
//					// データセット
//					tmpCompressData->data[tmpCompressData->dataPos] =
//						(float)tmpCompressData->compressingData / (float)tmpCompressData->compressBlock;
//
//					// 0に戻る
//					tmpCompressData->compressingData = NULL;
//					// カウント
//					tmpCompressData->dataPos++;
//				}
//				// カウント
//				dataCnt--;
//			}
//			else
//			{
//				soundResource->isWaveUpdate = false;
//				soundResource->isCompressed = true;
//				tmpCompressData->readPos = NULL;
//				tmpCompressData->dataPos = NULL;
//				tmpCompressData->compressingData = NULL;
//				tmpCompressData->compressBlock = NULL;
//				tmpCompressData->endTime = timeGetTime();
//				break;
//			}
//		}
//	}
//	return;
//}