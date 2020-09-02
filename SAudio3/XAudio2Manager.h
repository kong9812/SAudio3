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
	bool isPlaying;							// 再生状態
	int effectCnt;							// エフェクトの個数
	std::list<XAPO_LIST> effectList;		// エフェクトのリスト
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
	HRESULT CreateVoiceResourceVoice(IXAudio2 *xAudio2, std::string voiceName, SoundResource soundResource);

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

	// ボイス詳細
	XAUDIO2_VOICE_DETAILS GetVoiceDetails(std::string voiceName);

	// 処理サンプリングの設定
	void SetProcessSampling(int _processSampling);

	// 処理サンプリングの取得
	int GetProcessSampling(void);

	// サブミックスボイスの作成
	IXAudio2SubmixVoice *CreateSubmixVoice(std::string voiceName);
	
	// サブミックスボイスの作成
	IXAudio2SourceVoice *CreateSourceVoice(std::string voiceName);

	// アウトプットボイスの設定
	HRESULT SetOutputVoice(std::string voiceName,
		std::map <std::string, XAUDIO2_SEND_DESCRIPTOR> sendDescriptorList, int sendCount);

private:
	SoundBase				*soundBase;							// サウンドベース

	IXAudio2				*XAudio2;							// XAudio2
	IXAudio2MasteringVoice	*XAudio2MasteringVoice;				// マスターボイス
	XAudio2EffectManager	*xAudio2EffectManager;				// エフェクトマネージャー
	std::map<std::string, VoiceResource> voiceResource;			// ボイスリソース

	int						processSampling;					// 処理サンプリング
	float					oldMasteringVoiceVolume;			// 重い対策
};