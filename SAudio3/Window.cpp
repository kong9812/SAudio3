//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Window.h"
#include "imguiManager.h"

//===================================================================================================================================
// ローカル変数
//===================================================================================================================================
Window* window = NULL;	// ウインドウプロシージャ用

//===================================================================================================================================
// ウインドウプロシージャ(直接に使えないから)
//===================================================================================================================================
LRESULT CALLBACK wndProc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK wndProc(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	return window->WndProc(wnd, msg, wparam, lparam);
}

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
Window::Window()
{
	// メモリリークの検出
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	// スレッド並行数
	unsigned int count = std::thread::hardware_concurrency();

	// リサイズフラグの初期化
	reSizeFlg = false;
}

//===================================================================================================================================
// 初期化
//===================================================================================================================================
HRESULT Window::Init(HINSTANCE hInstance)
{
	// プロシージャ用
	window = this;

	// ウィンドウクラスの登録
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = (WNDPROC)wndProc;  //  ウィンドウプロシージャの登録
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;  //  アプリケーションインスタンス
	wcex.hIcon = LoadIcon(NULL, MAKEINTRESOURCE(IDI_APPLICATION));
	wcex.hCursor = LoadCursor(NULL, MAKEINTRESOURCE(IDC_ARROW));
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = NULL;
	wcex.lpszClassName = APP_NAME;  //  ウィンドウクラス名
	wcex.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	RegisterClassEx(&wcex);

	//  ウィンドウの生成
	hWnd = CreateWindowEx(
		WS_EX_OVERLAPPEDWINDOW,
		wcex.lpszClassName,
		APP_NAME,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		0,
		852,
		480,
		NULL,
		NULL,
		hInstance,
		NULL);

	if (!hWnd)
		return FALSE;
}

//===================================================================================================================================
// ウインドウプロシージャ
//===================================================================================================================================
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK Window::WndProc(HWND hWnd, UINT message, WPARAM _wParam, LPARAM _lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, message, _wParam, _lParam))
		return true;

	switch (message)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_KEYDOWN:case WM_SYSKEYDOWN:
		if (_wParam == VK_ESCAPE)
		{
			PostQuitMessage(0);
		}
		break;
	case WM_SIZE:
		reSizeFlg = true;
		wParam = _wParam;
		lParam = _lParam;
		break;
	default:
		return DefWindowProc(hWnd, message, _wParam, _lParam);
	}

	return DefWindowProc(hWnd, message, _wParam, _lParam);
}

//===================================================================================================================================
// ウインドの表示
//===================================================================================================================================
void Window::ShowWnd(int nCmdShow)
{
#if 0
	// ウィンドウサイズの調整
	GetWindowRect(hWnd, &bounds);
	GetClientRect(hWnd, &client);
	MoveWindow(hWnd, bounds.left, bounds.top,
		852 * 2 - client.right,
		480 * 2 - client.bottom,
		false);
#endif

	// ウィンドウの再表示
	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);
}

//===================================================================================================================================
// ウインドハンドルの取得
//===================================================================================================================================
HWND Window::GetWindowHwnd()
{
	return hWnd;
}

//===================================================================================================================================
// リサイズフラグの取得
//===================================================================================================================================
bool Window::GetReSizeFlg()
{
	return reSizeFlg;
}
//===================================================================================================================================
// リサイズフラグの設定
//===================================================================================================================================
void Window::SetReSizeFlg(bool _reSizeFlg)
{
	reSizeFlg = _reSizeFlg;
}

//===================================================================================================================================
// リサイズデータの取得(wParam)
//===================================================================================================================================
WPARAM Window::GetwParam()
{
	return wParam;
}

//===================================================================================================================================
// リサイズデータの取得(lParam)
//===================================================================================================================================
LPARAM Window::GetlParam()
{
	return lParam;
}