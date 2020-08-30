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
	int		fadeInPos;
	int		fadeOutPos;
	int		fadeInMs;
	int		fadeOutMs;
};

struct Mixer_Data
{
	std::list<Mixer_Resource> mixerResource;	// ミクサーリソース
	std::list<Mixer_Parameter> mixerParameter;		// ミクサーパラメーター
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class ImGuiMixerManager
{
public:
	ImGuiMixerManager(TextureBase *_textureBase);
	~ImGuiMixerManager();

	// 利用するサウンドの設置
	void SetMixerResource(std::string soundName, bool addUse);

	// ミクサーパネル
	void MixerPanel(bool *showMixerPanael);

private:
	Mixer_Data	mixerData;		// ミクサーデータ
	TextureBase	*textureBase;	// テクスチャベース

	// ミクサーパラメーターの作成
	Mixer_Parameter CreateMixerParameter(Mixer_Resource mixResourceData);

	// [パーツ]再生プレイヤー
	void MixerPartPlayer(std::list<Mixer_Parameter>::iterator mixerParameter, float buttomSize);

	// [パーツ]削除ボタン
	void MixerPartDelete(bool deleteButton);

	// [パーツ]ミクサー
	void MixerPartMixer(std::list<Mixer_Parameter>::iterator mixerParameter);
};