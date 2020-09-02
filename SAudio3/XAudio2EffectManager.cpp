//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "XAudio2EffectManager.h"
#include "SAudio3FadeXapo.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
XAudio2EffectManager::XAudio2EffectManager()
{
	XApo = new IUnknown*[XAPO_LIST::XAPO_MAX];
	XApo[XAPO_LIST::XAPO_FADE] = (IXAPO *)new SAudio3FadeXapo();
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
XAudio2EffectManager::~XAudio2EffectManager()
{
	// XAPOの終了処理
	for (int i = 0; i < XAPO_LIST::XAPO_MAX; i++)
	{
		SAFE_RELEASE(XApo[i])
	}
	SAFE_DELETE(XApo)
}

//===================================================================================================================================
// エフェクトの設置・解除
//===================================================================================================================================
HRESULT XAudio2EffectManager::SetXapoEffect(IXAudio2SubmixVoice *submixVoice, XAPO_LIST xapoID,
	int effectCnt, std::list<XAPO_LIST> effectList, bool isUse)
{
	bool isInit = false;	// 初期化状態
	int effectID = NULL;	// エフェクトID
	if (effectList.size() > NULL)
	{
		// エフェクトリスト検索
		for (auto _effectID : effectList)
		{
			// 該当するエフェクトが存在したら
			if (_effectID == xapoID)
			{
				isInit = true;			// 既に初期化されている
				effectID = _effectID;	// コピー
			}
		}
	}
	if (isUse)
	{
		// エフェクト初期化されていない
		if (!isInit)
		{
			XAUDIO2_EFFECT_DESCRIPTOR	effectDescriptor = { NULL };	// エフェクトディスクリプタ
			XAUDIO2_EFFECT_CHAIN		chain = { NULL };				// エフェクトチェン

			XAUDIO2_VOICE_DETAILS		voiceDetails = { NULL };		// ボイス詳細
			submixVoice->GetVoiceDetails(&voiceDetails);				// ボイス詳細の取得

			// エフェクトディスクリプタの初期化
			effectDescriptor.pEffect = XApo[xapoID];
			effectDescriptor.InitialState = isUse;
			effectDescriptor.OutputChannels = voiceDetails.InputChannels;

			// エフェクトチェンの初期化
			chain.EffectCount = effectCnt;
			chain.pEffectDescriptors = &effectDescriptor;

			// ソースボイスとの接続
			submixVoice->SetEffectChain(&chain);
		}
		else
		{
			// エフェクトの有効化
			submixVoice->EnableEffect(effectID);
		}
	}
	else
	{
		// エフェクトの無効化
		submixVoice->DisableEffect(effectID);
	}

	return S_OK;
}