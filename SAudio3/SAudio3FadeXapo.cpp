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
	XAPOASSERT(tmpParameters->fadeInPosMs <= tmpParameters->fadeOutPosMs);
	XAPOASSERT(tmpParameters->fadeInMs + tmpParameters->fadeOutMs <= tmpParameters->allSampling);

	// フェイドイン
	if (tmpParameters->fadeInMs != NULL)
	{
		// サンプリング変換
		fadeAddVolume = 1.0f / MS_TO_SAMPLING(tmpParameters->fadeInMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
		fadeInPosSampling = MS_TO_SAMPLING(tmpParameters->fadeInPosMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
		fadeInSampling = MS_TO_SAMPLING(tmpParameters->fadeInMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	}
	// フェイドアウト
	if (tmpParameters->fadeOutMs != NULL)
	{
		// サンプリング変換
		fadeMinusVolume = 1.0f / MS_TO_SAMPLING(tmpParameters->fadeOutMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
		fadeOutPosSampling = MS_TO_SAMPLING(tmpParameters->fadeOutPosMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
		fadeOutSampling = MS_TO_SAMPLING(tmpParameters->fadeOutMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	}
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

				float currentVolume = 1.0f;


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