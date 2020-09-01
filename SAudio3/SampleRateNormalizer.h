#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "SoundBase.h"

//===================================================================================================================================
// クラス
//===================================================================================================================================
class SampleRateNormalizer
{
public:
	SampleRateNormalizer() {};
	~SampleRateNormalizer() {};

	// サンプリング周波数の正規化
	short *SetSampleRate(SoundResource *soundResource, int sampleRate);
};