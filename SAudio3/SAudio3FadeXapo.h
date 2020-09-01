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
	int		fadeInPos;
	int		fadeOutPos;
	int		fadeInMs;
	int		fadeOutMs;
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class __declspec(uuid("{3C667D8D-4BA9-487F-8944-57419AB69909}"))SAudio3FadeXapo : public CXAPOParametersBase
{
public:
	SAudio3FadeXapo();
	~SAudio3FadeXapo() {};

	// Processに入る前の最終調整
	// 入力フォーマットと出力フォーマットの設定
	STDMETHOD(LockForProcess)
		(UINT32 inputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pInputLockedParameters,
			UINT32 outputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pOutputLockedParameters)
		override;

	// パラメーターの最終チェック
	STDMETHOD_(void, Process)
		(UINT32 inputProcessParameterCount,
			const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
			UINT32 outputProcessParameterCount,
			XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
			BOOL isEnabled)
		override;
	
	// エフェクト処理
	virtual void OnSetParameters
	(const void* pParameters, UINT32 ParameterByteSize);

private:
	// エフェクターの情報・特性
	XAPO_REGISTRATION_PROPERTIES registrationProperties;
	
	// フォーマット
	WAVEFORMATEX inputFormat;
	WAVEFORMATEX outputFormat;

	// 再生位置
	int playPos;

	// パラメータ
	SAudio3FadeParameter xapoParameter[3];
};