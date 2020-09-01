#pragma once
//===================================================================================================================================
// ライブラリフラグ
//===================================================================================================================================
#define XAUDIO2_HELPER_FUNCTIONS

//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include <xaudio2.h>
#include "Main.h"
#include "SoundBase.h"
#include "XAudio2EffectManager.h"

//===================================================================================================================================
// マクロ定義
//===================================================================================================================================
#define SAFE_DESTROY_VOICE(p)			if(p){  (p)->DestroyVoice(); p = NULL; }

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
namespace xAudioManagerNS
{
	const float minVolume = 0.0f;
	const float maxVolume = 1.0f;
	const float overVolume = 10.0f;
}

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct VoiceResource
{
	bool isPlaying;
	IXAudio2SourceVoice *sourceVoice;
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class XAudio2Manager
{
public:
	XAudio2Manager(SoundBase *_soundBase);
	~XAudio2Manager();

	// マスターボイスの作成
	IXAudio2MasteringVoice	*CreateMasterVoice(IXAudio2 *xAudio2);

	// ボイスリソースの作成
	void CreateVoiceResourceVoice(IXAudio2 *xAudio2, std::string voiceName, SoundResource soundResource);

	// ソースボイスの再生・一時停止
	void PlayPauseSourceVoice(IXAudio2 *xAudio2, std::string voiceName);

	// マスターボイスボリュームの取得
	float GetMasteringVoiceVolumeLevel(void);
	// マスターボイスボリュームの調整
	HRESULT SetMasteringVoiceVolumeLevel(float);

	// 再生状態
	bool GetIsPlaying(std::string voiceName);

	// ボイス状態
	XAUDIO2_VOICE_STATE GetVoiceState(std::string voiceName);

private:
	SoundBase				*soundBase;							// サウンドベース

	IXAudio2				*XAudio2;							// XAudio2
	IXAudio2MasteringVoice	*XAudio2MasteringVoice;				// マスターボイス
	XAudio2EffectManager	*xAudio2EffectManager;				// エフェクトマネージャー
	std::map<std::string, VoiceResource> voiceResource;			// ボイスリソース

	float					oldMasteringVoiceVolume;			// 重い対策
};