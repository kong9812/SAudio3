//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "SAudio3FadeXapo.h"

//===================================================================================================================================
// エフェクターの情報・特性
//===================================================================================================================================
XAPO_REGISTRATION_PROPERTIES SAudio3FadeXapo::registrationProperties = {
	__uuidof(SAudio3FadeXapo),
	L"SAudio3FadeXapo",
	L"Copyright (C)2020 CHOI YAU KONG",
	1,
	1,
	XAPOBASE_DEFAULT_FLAG,
	1, 1, 1, 1 };

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
SAudio3FadeXapo::SAudio3FadeXapo() :CXAPOParametersBase(&registrationProperties,
(BYTE *)xapoParameter, sizeof(SAudio3FadeParameter), false)
{
	samplingCnt = NULL;		// サンプリングカウンター(処理位置)
	fadeAddVolume = NULL;	// [フェイドイン]1サンプリング当たりのボリューム
	fadeMinusVolume = NULL;	// [フェイドアウト]1サンプリング当たりのボリューム
}

//===================================================================================================================================
// ①入力フォーマットと出力フォーマットの設定
//===================================================================================================================================
HRESULT SAudio3FadeXapo::LockForProcess
(UINT32 inputLockedParameterCount,
	const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pInputLockedParameters,
	UINT32 outputLockedParameterCount,
	const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pOutputLockedParameters)
{
	const HRESULT hr = CXAPOParametersBase::LockForProcess(
		inputLockedParameterCount,
		pInputLockedParameters,
		outputLockedParameterCount,
		pOutputLockedParameters);

	if (SUCCEEDED(hr))
	{
		// 0番目から取り出す
		memcpy(&inputFormat, pInputLockedParameters[0].pFormat, sizeof(inputFormat));
		memcpy(&outputFormat, pOutputLockedParameters[0].pFormat, sizeof(outputFormat));
	}

	return hr;
}

//===================================================================================================================================
// ②パラメーターチェック
//===================================================================================================================================
void SAudio3FadeXapo::SetParameters
(const void* pParameters, UINT32 ParameterByteSize)
{
	// サイズチェックのみ
	if (ParameterByteSize == sizeof(SAudio3FadeParameter))
	{
		return CXAPOParametersBase::SetParameters(pParameters, ParameterByteSize);
	}

	return;
}

//===================================================================================================================================
// ③Process直前の初期化・パラメーターの最終チェック
//===================================================================================================================================
void SAudio3FadeXapo::OnSetParameters
(const void* pParameters, UINT32 ParameterByteSize)
{
	SAudio3FadeParameter *tmpParameters = ((SAudio3FadeParameter *)pParameters);

	// チェックリスト
	XAPOASSERT(sizeof(SAudio3FadeParameter) > 0);
	XAPOASSERT(pParameters != NULL);
	XAPOASSERT(ParameterByteSize == sizeof(SAudio3FadeParameter));
	XAPOASSERT(tmpParameters->fadeInStartMs != tmpParameters->fadeInEndMs);
	XAPOASSERT(tmpParameters->fadeOutStartMs != tmpParameters->fadeOutEndMs);
	XAPOASSERT(tmpParameters->fadeInStartMs < tmpParameters->fadeOutStartMs);
	XAPOASSERT(tmpParameters->fadeInStartMs < tmpParameters->fadeOutEndMs);
	XAPOASSERT(tmpParameters->fadeInEndMs < tmpParameters->fadeOutStartMs);
	XAPOASSERT(tmpParameters->fadeInEndMs < tmpParameters->fadeOutEndMs);

	// フェイドイン
	fadeInStartSampling = MS_TO_SAMPLING(tmpParameters->fadeInStartMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeInEndSampling = MS_TO_SAMPLING(tmpParameters->fadeInEndMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeAddVolume = 1.0f / ((float)fadeInEndSampling - (float)fadeInStartSampling);
	// フェイドアウト
	fadeOutStartSampling = MS_TO_SAMPLING(tmpParameters->fadeOutStartMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeOutEndSampling = MS_TO_SAMPLING(tmpParameters->fadeOutEndMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeMinusVolume = 1.0f / ((float)fadeOutEndSampling - (float)fadeOutStartSampling);
}

//===================================================================================================================================
// ④エフェクト処理
//===================================================================================================================================
void SAudio3FadeXapo::Process
(UINT32 inputProcessParameterCount,
	const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
	UINT32 outputProcessParameterCount,
	XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
	BOOL isEnabled)
{
	// 仮のパラメータ構造体 = 使用するパラメータのポインたー
	SAudio3FadeParameter *tmpParameter = (SAudio3FadeParameter *)BeginProcess();

	if (isEnabled)
	{
		if (pInputProcessParameters->BufferFlags != XAPO_BUFFER_FLAGS::XAPO_BUFFER_SILENT)
		{
			for (int i = 0; i < ((int)pInputProcessParameters->ValidFrameCount * inputFormat.nChannels); i++)
			{
				// 入出力のバッファ
				float *inputBuf = (float *)pInputProcessParameters->pBuffer;
				float *outputBuf = (float *)pOutputProcessParameters->pBuffer;

				// フェイドイン処理
				if ((samplingCnt >= fadeInStartSampling) &&
					(samplingCnt <= fadeInEndSampling))
				{
					// フェイド中の位置
					int fadeIdx = samplingCnt - fadeInStartSampling;

					// ボリューム計算
					float volume = (fadeAddVolume)*fadeIdx;
					outputBuf[i] = inputBuf[i] * volume;
				}
				else if ((samplingCnt >= fadeOutStartSampling) &&
						(samplingCnt <= fadeOutEndSampling))
				{
					// フェイド中の位置
					int fadeIdx = samplingCnt - fadeOutStartSampling;

					// ボリューム計算
					float volume = 1.0f - ((fadeMinusVolume)*fadeIdx);
					outputBuf[i] = inputBuf[i] * volume;
				}

				// サンプリングカウンター(処理位置)
				samplingCnt++;
				if (samplingCnt >= tmpParameter->allSampling)
				{
					samplingCnt = 0;
				}
			}
		}
	}
	// エンドプロセス
	EndProcess();
}

//===================================================================================================================================
// α送信用
//===================================================================================================================================
void SAudio3FadeXapo::GetParameters
(void* pParameters, UINT32 ParameterByteSize)
{
	// 処理進捗
	if (ParameterByteSize == sizeof(int))
	{
		*(int *)pParameters = samplingCnt;
	}
}