//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "imguiManager.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
ImGuiManager::ImGuiManager(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext	*deviceContext)
{
	// バージョンチェック
	IMGUI_CHECKVERSION();

	// [ImGui]コンテクストの作成
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

#if USE_IMGUI_DOCKING
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// ドッキングの使用許可
#endif

	// ダークモード
	ImGui::StyleColorsClassic();

	// [ImGui]win32の初期化
	if (!ImGui_ImplWin32_Init(hWnd))
	{
		// エラーメッセージ
		MessageBox(NULL, errorNS::ImGuiWin32InitError, APP_NAME, (MB_OK | MB_ICONERROR));

		// 強制終了
		PostQuitMessage(0);
		return;
	}

	// [ImGui]初期化
	if (!ImGui_ImplDX11_Init(device, deviceContext))
	{
		// エラーメッセージ
		MessageBox(NULL, errorNS::ImGuiInitError, APP_NAME, (MB_OK | MB_ICONERROR));

		// 強制終了
		PostQuitMessage(0);
		return;
	}

	// ImGuiフラグの初期化
	showMainPanel = true;
	showPlayerPanel = true;
}

//===================================================================================================================================
// [ImGui]新しいフレームの作成
//===================================================================================================================================
void ImGuiManager::CreateNewFrame()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
ImGuiManager::~ImGuiManager()
{
	// [ImGui]終了処理
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

//===================================================================================================================================
// [ImGui]リサイズ
//===================================================================================================================================
void ImGuiManager::ReSize(LONG right, LONG bottom)
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(right, bottom));
}

//===================================================================================================================================
// [ImGui]メインパネル(リサイズ)
//===================================================================================================================================
void ImGuiManager::ShowPanel(bool reSize, RECT mainPanelSize)
{
	// リサイズ
	if (reSize)
	{
		ReSize(mainPanelSize.right, mainPanelSize.bottom);
	}

	// メインパネル
	MainPanel();
}

//===================================================================================================================================
// [ImGui]メインパネル
//===================================================================================================================================
void ImGuiManager::ShowPanel()
{
	// メインパネル
	MainPanel();
}

//===================================================================================================================================
// メインパネル
//===================================================================================================================================
void ImGuiManager::MainPanel()
{
	// メインパネル
	if (showMainPanel)
	{
		ImGui::SetNextWindowBgAlpha(0.7f);
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("SAudio3", &showMainPanel,
			ImGuiWindowFlags_::ImGuiWindowFlags_MenuBar |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoBringToFrontOnFocus);

		// メニューバー
		MenuBar();

		// テスト文字の表示
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		// ドッキング
		ImGuiID dockspaceID = ImGui::GetID("MainPanelDockSpace");
		ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
		ImGui::End();

	}

	// 再生パネル
	PlayerPanel();
}

//===================================================================================================================================
// メニューバー
//===================================================================================================================================
void ImGuiManager::MenuBar()
{
	if (ImGui::BeginMenuBar())
	{
		// ファイル操作
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("New")){}
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Window"))
		{
			ImGui::MenuItem("Player", "", &showPlayerPanel);
			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}
}

//===================================================================================================================================
// 再生パネル
//===================================================================================================================================
void ImGuiManager::PlayerPanel()
{
	// 再生パネル
	if (showPlayerPanel)
	{
		ImGui::Begin("Player Panel", &showPlayerPanel, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);

		// テスト文字の表示
		ImGui::Text("player");

		ImGui::End();
	}
}