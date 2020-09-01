//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Window.h"
#include "imguiManager.h"
#include "AppProc.h"

//===================================================================================================================================
// プロトタイプ宣言
//===================================================================================================================================
WPARAM MessageLoop(Window *window, MSG msg);	// メッセージループ

//===================================================================================================================================
// WinMain
//===================================================================================================================================
INT APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	// ウインド
	Window *window = new Window;

	// ウインドの初期化
	if (!window->Init(hInstance))
		return FALSE;

	// ウィンドウの再表示
	window->ShowWnd(nCmdShow);

	// メッセージループ&終了処理
	MSG msg = { 0 };
	ZeroMemory(&msg, sizeof(msg));
	return (int)MessageLoop(window, msg);
}

//===================================================================================================================================
// メッセージループ
//===================================================================================================================================
WPARAM MessageLoop(Window *window, MSG msg)
{
	// ウインドハンドル
	HWND hWnd[WINDOWS_ID::MAX_WINDOWS] =
	{ window->GetWindowHwnd(WINDOWS_ID::MAIN_WINDOWS),
		window->GetWindowHwnd(WINDOWS_ID::SUB_WINDOWS) };
	// DX11ハンドル
	DirectX11 *directX11[WINDOWS_ID::MAX_WINDOWS] = { nullptr };
	
	for (int i = 0; i < WINDOWS_ID::MAX_WINDOWS; i++)
	{
		// DX11の初期化
		directX11[i] = new DirectX11(hWnd[i]);
		if (!directX11[i]->Init())
		{
			// エラーメッセージ
			MessageBox(NULL, errorNS::DXInitError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

			// 強制終了
			PostQuitMessage(0);
		}
	}

	// アプリプロセスの初期化
	AppProc *appProc = new AppProc(hWnd[WINDOWS_ID::MAIN_WINDOWS],
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetDevice(),
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetDeviceContext(),
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetSwapChain(),
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetRenderTargetView());

	// ランチャーの終了時間(1秒)
	int launcherEndTime = timeGetTime() + 1000;
	bool isLauncher = true;

	// メッセージループ
	while (true)
	{
		// リサイズ
		if (window->GetReSizeFlg() == true)
		{
			if (directX11[WINDOWS_ID::MAIN_WINDOWS]->GetDevice() != NULL && window->GetwParam() != SIZE_MINIMIZED)
			{
				directX11[WINDOWS_ID::MAIN_WINDOWS]->ReSize(window->GetlParam());
				window->SetReSizeFlg(false);
				appProc->ReSize(true);
			}
		}

		// メッセージ解析
		if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				break;
			}
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			if (isLauncher)
			{
				if (timeGetTime() >= launcherEndTime)
				{
					window->CloseWindow(WINDOWS_ID::SUB_WINDOWS);
					window->ShowWndAgain(WINDOWS_ID::MAIN_WINDOWS);
					isLauncher = false;
				}
			}
			appProc->Update(hWnd[WINDOWS_ID::MAIN_WINDOWS]);	// 更新処理
			appProc->Draw(directX11[WINDOWS_ID::MAIN_WINDOWS]->GetRenderTargetView());	// 描画処理
		}
	}

	// 終了処理
	SAFE_DELETE(appProc)
	for (int i = 0; i < WINDOWS_ID::MAX_WINDOWS; i++)
	{
		SAFE_DELETE(directX11[i])
	}

	return msg.wParam;
}