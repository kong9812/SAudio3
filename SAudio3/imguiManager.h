#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "TextureBase.h"
#include "SoundBase.h"
#include "XAudio2Manager.h"
#include "ImGuiPlotManager.h"
#include "ImGuiMixerManager.h"

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
#include "imGui/docking/imgui_internal.h"
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
	const float masteringVoiceVolumeSliderWidth = 400;
	const ImVec2 soundBasePanelProgressBarSize = ImVec2(-1.0f, 5.0f);
}

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct Performance_Record
{
	int cpuNum;
	SYSTEM_INFO systeminfo;
	MEMORYSTATUSEX memoryStatusEx;
	PDH_HQUERY pdhHquery;
	PDH_HCOUNTER cpuCounter;
	PDH_HCOUNTER memoryCounter;
	float cpuUsage;
	float memoryUsage;
	int timeCnt;
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class ImGuiManager
{
public:
	ImGuiManager(HWND hWnd, ID3D11Device *device,
		ID3D11DeviceContext *deviceContext, TextureBase *textureBase,
		SoundBase *soundBase, XAudio2Manager *xAudio2Manager);
	~ImGuiManager();

	// [ImGui]新しいフレームの作成
	void CreateNewFrame();

	// [ImGui]ヘルプマーク
	void HelpMarker(const char* desc);

	// [ImGui]パネルの表示
	void ShowPanel(bool reSize, RECT mainPanelSize);
	void ShowPanel();

private:
	TextureBase			*textureBase;		// テクスチャベース
	SoundBase			*soundBase;			// サウンドベース
	XAudio2Manager		*xAudio2Manager;	// XAudio2マネージャー
	ImGuiPlotManager	*imGuiPlotManager;	// [ImGui]プロットマネージャー
	ImGuiMixerManager	*imGuiMixerManager;	// [ImGui]ミクサーマネージャー

	bool showMainPanel;					// [ImGuiフラグ]メインパネル
	bool showPlayerPanel;				// [ImGuiフラグ]再生パネル
	bool showSoundBasePanel;			// [ImGuiフラグ]サウンドベースパネル
	bool showMixerPanel;				// [ImGuiフラグ]サウンドベースパネル
	bool isPlaying;						// [プレイヤーパネル]再生中??
	bool isMasteringVoiceVolumeOver1;	// マスターボイスのボリュームが1を超えられる?

	Performance_Record performanceRecord;	// パフォーマンス記録

	// [ImGui]リサイズ
	void ReSize(LONG right, LONG bottom);

	// メインパネル
	void MainPanel();

	// パフォーマンスビューアの初期化
	void PerformanceViewerInit();

	// パフォーマンスビューア
	void PerformanceViewer();

	// メニューバー
	void MenuBar();

	// 再生パネル
	void PlayerPanel();

	// マスターボイスボリューム
	void MasteringVoiceVolumePanel();

	// サウンドベースパネル
	void SoundBasePanel();
};