#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "TextureBase.h"
#include "SoundBase.h"
#include "XAudio2Manager.h"

//===================================================================================================================================
// ビルドスイッチ
//===================================================================================================================================
// ドッキングの使用状況(DONT SWITCH TO FALSE!!!!!)
// trueのみ
#define USE_IMGUI_DOCKING (true)
#if USE_IMGUI_DOCKING
#include "imGui/docking/imgui.h"
#include "imGui/docking/imgui_impl_win32.h"
#include "imGui/docking/imgui_impl_dx11.h"
#else
#include "imGui/imgui.h"
#include "imGui/imgui_impl_win32.h"
#include "imGui/imgui_impl_dx11.h"
#endif

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
namespace imGuiManagerNS
{
	const ImVec2 buttonSize = ImVec2(25, 25);
}

//===================================================================================================================================
// クラス
//===================================================================================================================================
class ImGuiManager
{
public:
	ImGuiManager(HWND hWnd, ID3D11Device *device, 
		ID3D11DeviceContext *deviceContext, TextureBase *textureBase,
		SoundBase *soundBase);
	~ImGuiManager();

	// [ImGui]新しいフレームの作成
	void CreateNewFrame();

	// [ImGui]パネルの表示
	void ShowPanel(bool reSize, RECT mainPanelSize);
	void ShowPanel();

private:
	TextureBase *textureBase;	// テクスチャベース
	SoundBase	*soundBase;		// サウンドベース

	bool showMainPanel;			// [ImGuiフラグ]メインパネル
	bool showPlayerPanel;		// [ImGuiフラグ]再生パネル
	bool showSoundBasePanel;	// [ImGuiフラグ]サウンドベースパネル

	bool isPlaying;				// [プレイヤーパネル]再生中??

	// [ImGui]リサイズ
	void ReSize(LONG right, LONG bottom);

	// メインパネル
	void MainPanel();

	// メニューバー
	void MenuBar();

	// 再生パネル
	void PlayerPanel();

	// サウンドベースパネル
	void SoundBasePanel();
};