#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "TextureBase.h"
#include "SoundBase.h"

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct Mixer_Resource
{
	std::string soundName;	// サウンド名
	int cnt;				// 利用回数
};

struct Mixer_Parameter
{
	std::string soundName;
	bool	isFade;
	bool	isPlaying;
	float	playingPos;
	float	fadeInPos;
	float	fadeOutPos;
	float	fadeInMs;
	float	fadeOutMs;
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
	ImGuiMixerManager(TextureBase *_textureBase, SoundBase *_soundBase);
	~ImGuiMixerManager();

	// 利用するサウンドの設置
	void SetMixerResource(std::string soundName, bool addUse);

	// ミクサーパネル
	void MixerPanel(bool *showMixerPanael);

private:
	TextureBase	*textureBase;	// テクスチャベース
	SoundBase	*soundBase;		// サウンドベース
	Mixer_Data	mixerData;		// ミクサーデータ

	// ミクサーパラメーターの作成
	Mixer_Parameter CreateMixerParameter(Mixer_Resource mixResourceData);

	// [パーツ]再生プレイヤー
	void MixerPartPlayer(std::list<Mixer_Parameter>::iterator mixerParameter, float buttomSize);

	// [パーツ]削除ボタン
	void MixerPartDelete(bool deleteButton);

	// [パーツ]ミクサー
	void MixerPartMixer(std::list<Mixer_Parameter>::iterator mixerParameter);
};