#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include <xapo.h>
#include <xapobase.h>
#include <xapofx.h>
#include <xaudio2.h>
#pragma comment(lib,"xapobase.lib")

#include "Main.h"

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
enum XAPO_LIST
{
	XAPO_FADE,
	XAPO_MAX
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class XAudio2EffectManager
{
public:
	XAudio2EffectManager();
	~XAudio2EffectManager();

	// フェイドの設置
	HRESULT SetXapoFade(IXAudio2SourceVoice *sourceVoice);

	// エフェクトの設置・解除
	HRESULT SetXapoEffect(IXAudio2SourceVoice *sourceVoice, XAPO_LIST xapoID,
		int effectCnt, std::list<XAPO_LIST> effectList, bool isUse);

private:
};