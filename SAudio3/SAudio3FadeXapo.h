#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct SAudio3FadeParameter
{
	long	allSampling;
	int		fadeInStartMs;	// [フェイドイン]フェイドの開始時間(ms)
	int		fadeInEndMs;	// [フェイドイン]フェイドの完了時間(ms)
	int		fadeOutStartMs;	// [フェイドアウト]フェイドの開始時間(ms)
	int		fadeOutEndMs;	// [フェイドアウト]フェイドの完了時間(ms)
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class __declspec(uuid("{3C667D8D-4BA9-487F-8944-57419AB69909}"))SAudio3FadeXapo : public CXAPOParametersBase
{
public:
	SAudio3FadeXapo();
	~SAudio3FadeXapo() {};

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

	// 送信用
	STDMETHOD_(void, GetParameters)
		(_Out_writes_bytes_(ParameterByteSize) void* pParameters, UINT32 ParameterByteSize)override;

private:
	// エフェクターの情報・特性
	static XAPO_REGISTRATION_PROPERTIES registrationProperties;
	
	// フォーマット
	WAVEFORMATEX inputFormat;
	WAVEFORMATEX outputFormat;

	int		samplingCnt;			// サンプリングカウンター(処理位置)
	float	fadeAddVolume;			// [フェイドイン]1サンプリング当たりのボリューム
	float	fadeMinusVolume;		// [フェイドアウト]1サンプリング当たりのボリューム
	int		fadeInStartSampling;	// [フェイドイン]フェイドの開始位置(サンプリング)
	int		fadeInEndSampling;		// [フェイドイン]フェイドの完了位置(サンプリング)
	int		fadeOutStartSampling;	// [フェイドアウト]フェイドの開始位置(サンプリング)
	int		fadeOutEndSampling;		// [フェイドアウト]フェイドの完了位置(サンプリング)

	// パラメータ
	SAudio3FadeParameter xapoParameter[3];
};