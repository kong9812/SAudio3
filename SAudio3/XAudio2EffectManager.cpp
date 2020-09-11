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
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
XAudio2EffectManager::~XAudio2EffectManager()
{
	//// XAPOの終了処理
	//SAFE_RELEASE(XApo)
}

//===================================================================================================================================
// フェードの設置
//===================================================================================================================================
HRESULT XAudio2EffectManager::SetXapoFade(IXAudio2SourceVoice *sourceVoice)
{
	XAUDIO2_EFFECT_DESCRIPTOR	effectDescriptor = { NULL };	// エフェクトディスクリプタ
	XAUDIO2_EFFECT_CHAIN		chain = { NULL };				// エフェクトチェン

	XAUDIO2_VOICE_DETAILS		voiceDetails = { NULL };		// ボイス詳細
	sourceVoice->GetVoiceDetails(&voiceDetails);				// ボイス詳細の取得

	// XAPOs
	IUnknown *XApo = (IXAPO *)new SAudio3FadeXapo();
	// エフェクトディスクリプタの初期化
	effectDescriptor.pEffect = XApo;
	effectDescriptor.InitialState = true;
	effectDescriptor.OutputChannels = voiceDetails.InputChannels;

	// エフェクトチェンの初期化
	chain.EffectCount = 1;
	chain.pEffectDescriptors = &effectDescriptor;

	// ソースボイスとの接続
	sourceVoice->SetEffectChain(&chain);

	// すぐぽい！(たぶん大丈夫…確認待ち)
	SAFE_RELEASE(XApo);

	return S_OK;
}

//===================================================================================================================================
// エフェクトの設置・解除
//===================================================================================================================================
HRESULT XAudio2EffectManager::SetXapoEffect(IXAudio2SourceVoice *sourceVoice, XAPO_LIST xapoID,
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
			sourceVoice->GetVoiceDetails(&voiceDetails);				// ボイス詳細の取得

			// エフェクトディスクリプタの初期化
			IUnknown *XApo = (IXAPO *)new SAudio3FadeXapo();;
			effectDescriptor.pEffect = XApo;
			effectDescriptor.InitialState = isUse;
			effectDescriptor.OutputChannels = voiceDetails.InputChannels;

			// エフェクトチェンの初期化
			chain.EffectCount = effectCnt;
			chain.pEffectDescriptors = &effectDescriptor;

			// ソースボイスとの接続
			sourceVoice->SetEffectChain(&chain);
		}
		else
		{
			// エフェクトの有効化
			sourceVoice->EnableEffect(effectID);
		}
	}
	else
	{
		// エフェクトの無効化
		sourceVoice->DisableEffect(effectID);
	}

	return S_OK;
}