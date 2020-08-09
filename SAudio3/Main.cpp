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
	HWND hWnd = window->GetWindowHwnd();

	// DX11の初期化
	DirectX11 *directX11 = new DirectX11(hWnd);
	if (!directX11->Init())
	{
		// エラーメッセージ
		MessageBox(NULL, errorNS::DXInitError, APP_NAME, (MB_OK | MB_ICONERROR));

		// 強制終了
		PostQuitMessage(0);
	}

	// アプリプロセスの初期化
	AppProc *appProc = new AppProc(hWnd,
		directX11->GetDevice(),
		directX11->GetDeviceContext(),
		directX11->GetSwapChain(),
		directX11->GetRenderTargetView());

	// メッセージループ
	while (true)
	{
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
			appProc->Update();	// 更新処理
			appProc->Draw(directX11->GetRenderTargetView());	// 描画処理
		}

		// リサイズ
		if (window->GetReSizeFlg() == true)
		{
			if (directX11->GetDevice() != NULL && window->GetwParam() != SIZE_MINIMIZED)
			{
				directX11->ReSize(window->GetlParam());
				window->SetReSizeFlg(false);
			}
		}
	}

	// 終了処理
	SAFE_DELETE(appProc)
	SAFE_DELETE(directX11)

	return msg.wParam;
}