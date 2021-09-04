#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// マクロ定義
//===================================================================================================================================
#define HS_TO_RADIAN_FREQUENCY(hs,sf)	(2.0f * M_PI * hs / sf)
#define S_HS_TO_RADIAN_FREQUENCY(hs,sf)	(hs / sf * 6.f)				// 簡易化

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
enum XAPO_FILTER_TYPE : int
{
	XFT_LowpassFilter,
	XFT_HighpassFilter,
	XFT_BandpassFilter,
	XFT_NotchFilter,
	XFT_MAX
};

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct SAudio3FilterParameter
{
	XAPO_FILTER_TYPE type;		// フィルターの種類
	int cutoffFreq;				// カットオフ周波数
	float Q;					// Q値
	float bandwidth;			// 帯域幅
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class __declspec(uuid("{d33db5ae-0d7b-11ec-82a8-0242ac130003}"))SAudio3FilterXapo : public CXAPOParametersBase
{
public:
	SAudio3FilterXapo();
	~SAudio3FilterXapo() {};

	// 入力フォーマットと出力フォーマットの設定
	STDMETHOD(LockForProcess)
		(UINT32 inputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pInputLockedParameters,
			UINT32 outputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pOutputLockedParameters)override;

	// パラメーターチェック
	STDMETHOD_(void, SetParameters)
		(_In_reads_bytes_(ParameterByteSize) const void* pParameters, UINT32 ParameterByteSize)override;

	// パラメーターの最終チェック
	virtual void OnSetParameters
	(const void* pParameters, UINT32 ParameterByteSize)override;

	// エフェクト処理
	STDMETHOD_(void, Process)
		(UINT32 inputProcessParameterCount,
			const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
			UINT32 outputProcessParameterCount,
			XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
			BOOL isEnabled)override;

private:
	// エフェクターの情報・特性
	static XAPO_REGISTRATION_PROPERTIES registrationProperties;

	// フォーマット
	WAVEFORMATEX inputFormat;
	WAVEFORMATEX outputFormat;

	// フィルタ係数
	float alpha;
	float omega;
	float a[3];
	float b[3];

	// 計算用
	float inputBackup[2];
	float outputBackup[2];
	
	// パラメータ
	SAudio3FilterParameter xapoParameter[3];

	// フィルター処理
	float BiQuadFilter(float input);		// BiQuad(双2次)フィルター
};