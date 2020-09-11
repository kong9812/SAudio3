#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "TextureBase.h"
#include "SoundBase.h"
#include "SAudio3FadeXapo.h"

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct Mixer_Resource
{
	//std::map <std::string, XAUDIO2_SEND_DESCRIPTOR>sendDescriptor;
	std::string soundName;	// サウンド名
	int cnt;				// 利用回数
};

struct Mixer_Parameter
{
	SAudio3FadeParameter sAudio3FadeParameter;		
	IXAudio2SourceVoice *XAudio2SourceVoice;	// テスト再生用
	std::string soundName;
	std::string parameterName;
	bool	isFade;
	bool	isPlaying;
	float	playingPos;
	int		maxSample;
	int		maxMs;
};

struct Mixer_Data
{
	std::list<Mixer_Resource> mixerResource;	// ミクサーリソース
	std::list<Mixer_Parameter> mixerParameter;	// ミクサーパラメーター
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class ImGuiMixerManager
{
public:
	ImGuiMixerManager(XAudio2Manager *_xAudio2Manager, TextureBase *_textureBase, SoundBase *_soundBase);
	~ImGuiMixerManager();

	// 利用するサウンドの設置
	void SetMixerResource(std::string soundName, bool addUse);

	// ミクサーパネル
	void MixerPanel(bool *showMixerPanael);

private:
	XAudio2Manager	*xAudio2Manager;// XAudio2マネジャー
	TextureBase		*textureBase;	// テクスチャベース
	SoundBase		*soundBase;		// サウンドベース
	Mixer_Data		mixerData;		// ミクサーデータ

	// ミクサーパラメーターの作成
	Mixer_Parameter CreateMixerParameter(Mixer_Resource mixResourceData);

	// [パーツ]再生プレイヤー
	void MixerPartPlayer(std::list<Mixer_Parameter>::iterator mixerParameter, float buttomSize);

	// [パーツ]削除ボタン
	bool MixerPartDelete(std::list<Mixer_Parameter>::iterator mixerParameter, bool deleteButton);

	// [パーツ]ミクサー
	void MixerPartMixer(std::list<Mixer_Parameter>::iterator mixerParameter);

	//// 送信ディスクリプタの作成・設置
	//void SetSendDescriptor(std::string mixerParameterName,
	//	std::list<Mixer_Resource>::iterator mixerResource, IXAudio2SubmixVoice *XAudio2SubmixVoice);
};