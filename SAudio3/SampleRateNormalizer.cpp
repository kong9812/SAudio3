//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "SampleRateNormalizer.h"
#include "cudaCalc.cuh"

//===================================================================================================================================
// サンプリング周波数の正規化
//===================================================================================================================================
short *SampleRateNormalizer::SetSampleRate(SoundResource *soundResource, int sampleRate)
{
	// GPU計算用クラス
	CUDA_CALC *cudaCalc = new CUDA_CALC;

	// [CUDA]正規化
	cudaCalc->normalizer(soundResource->data, soundResource->size, soundResource->waveFormatEx.nChannels,
		soundResource->waveFormatEx.nSamplesPerSec, sampleRate, 2.0f);

	// 後片付け
	SAFE_DELETE(cudaCalc)
	
	return nullptr;
}