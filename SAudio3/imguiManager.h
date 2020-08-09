#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"

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
// クラス
//===================================================================================================================================
class ImGuiManager
{
public:
	ImGuiManager(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext *deviceContext);
	~ImGuiManager();

	// [ImGui]新しいフレームの作成
	void CreateNewFrame();

	// [ImGui]パネルの表示
	void ShowPanel(bool reSize, RECT mainPanelSize);
	void ShowPanel();

private:

	bool showMainPanel;		// [ImGuiフラグ]メインパネル
	bool showPlayerPanel;	// [ImGuiフラグ]再生パネル

	// [ImGui]リサイズ
	void ReSize(LONG right, LONG bottom);

	// メインパネル
	void MainPanel();

	// メニューバー
	void MenuBar();

	// 再生パネル
	void PlayerPanel();
};