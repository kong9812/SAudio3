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
LRESULT CALLBACK mainWndProc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK mainWndProc(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	return window->WndProc1(wnd, msg, wparam, lparam);
}
LRESULT CALLBACK subWndProc2(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK subWndProc2(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	return window->WndProc2(wnd, msg, wparam, lparam);
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
	wcex[WINDOWS_ID::MAIN_WINDOWS].cbSize = sizeof(WNDCLASSEX);
	wcex[WINDOWS_ID::MAIN_WINDOWS].style = CS_HREDRAW | CS_VREDRAW;
	wcex[WINDOWS_ID::MAIN_WINDOWS].lpfnWndProc = (WNDPROC)mainWndProc;  //  ウィンドウプロシージャの登録
	wcex[WINDOWS_ID::MAIN_WINDOWS].cbClsExtra = 0;
	wcex[WINDOWS_ID::MAIN_WINDOWS].cbWndExtra = 0;
	wcex[WINDOWS_ID::MAIN_WINDOWS].hInstance = hInstance;  //  アプリケーションインスタンス
	wcex[WINDOWS_ID::MAIN_WINDOWS].hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
	wcex[WINDOWS_ID::MAIN_WINDOWS].hCursor = LoadCursor(NULL, MAKEINTRESOURCE(IDC_ARROW));
	wcex[WINDOWS_ID::MAIN_WINDOWS].hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex[WINDOWS_ID::MAIN_WINDOWS].lpszMenuName = NULL;
	wcex[WINDOWS_ID::MAIN_WINDOWS].lpszClassName = MAIN_APP_NAME;  //  ウィンドウクラス名
	wcex[WINDOWS_ID::MAIN_WINDOWS].hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
	RegisterClassEx(&wcex[WINDOWS_ID::MAIN_WINDOWS]);

	wcex[WINDOWS_ID::SUB_WINDOWS].cbSize = sizeof(WNDCLASSEX);
	wcex[WINDOWS_ID::SUB_WINDOWS].style = CS_HREDRAW | CS_VREDRAW;
	wcex[WINDOWS_ID::SUB_WINDOWS].lpfnWndProc = (WNDPROC)subWndProc2;  //  ウィンドウプロシージャの登録
	wcex[WINDOWS_ID::SUB_WINDOWS].cbClsExtra = 0;
	wcex[WINDOWS_ID::SUB_WINDOWS].cbWndExtra = 0;
	wcex[WINDOWS_ID::SUB_WINDOWS].hInstance = hInstance;  //  アプリケーションインスタンス
	wcex[WINDOWS_ID::SUB_WINDOWS].hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(NULL));
	wcex[WINDOWS_ID::SUB_WINDOWS].hCursor = LoadCursor(NULL, MAKEINTRESOURCE(NULL));
	wcex[WINDOWS_ID::SUB_WINDOWS].hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex[WINDOWS_ID::SUB_WINDOWS].lpszMenuName = NULL;
	wcex[WINDOWS_ID::SUB_WINDOWS].lpszClassName = SUB_APP_NAME;  //  ウィンドウクラス名
	wcex[WINDOWS_ID::SUB_WINDOWS].hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(NULL));
	RegisterClassEx(&wcex[WINDOWS_ID::SUB_WINDOWS]);

	//  ウィンドウの生成
	hWnd[WINDOWS_ID::MAIN_WINDOWS] = CreateWindowEx(
		WS_EX_OVERLAPPEDWINDOW,
		wcex[WINDOWS_ID::MAIN_WINDOWS].lpszClassName,
		MAIN_APP_NAME,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		0,
		windowNS::WINDOW_MAIN_WIDTH,
		windowNS::WINDOW_MAIN_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);
	hWnd[WINDOWS_ID::SUB_WINDOWS] = CreateWindowEx(
		WS_EX_OVERLAPPEDWINDOW,
		wcex[WINDOWS_ID::SUB_WINDOWS].lpszClassName,
		SUB_APP_NAME,
		WS_POPUPWINDOW,
		GetSystemMetrics(SM_CXSCREEN) / 2 - windowNS::WINDOW_SUB_WIDTH / 2,
		GetSystemMetrics(SM_CYSCREEN) / 2 - windowNS::WINDOW_SUB_HEIGHT / 2,
		windowNS::WINDOW_SUB_WIDTH,
		windowNS::WINDOW_SUB_HEIGHT,
		hWnd[WINDOWS_ID::MAIN_WINDOWS],
		NULL,
		hInstance,
		NULL);

	if (!hWnd[WINDOWS_ID::MAIN_WINDOWS] || !hWnd[WINDOWS_ID::SUB_WINDOWS])
		return FALSE;

	return TRUE;
}

//===================================================================================================================================
// ウインドウプロシージャ
//===================================================================================================================================
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK Window::WndProc1(HWND hWnd, UINT message, WPARAM _wParam, LPARAM _lParam)
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
		// リサイズデータ
		reSizeFlg = true;
		wParam = _wParam;
		lParam = _lParam;
		break;
	default:
		return DefWindowProc(hWnd, message, _wParam, _lParam);
	}

	return DefWindowProc(hWnd, message, _wParam, _lParam);
}
LRESULT CALLBACK Window::WndProc2(HWND hWnd, UINT message, WPARAM _wParam, LPARAM _lParam)
{
	int x = 0;
	switch (message)
	{
	case WM_CREATE:
		//アイコンロード
		icon = (HICON)LoadImage(NULL,
			"icon.ico",
			IMAGE_ICON,
			0,
			0,
			LR_LOADFROMFILE);
		if (icon == NULL)
		{
			MessageBox(NULL, "情報を取得できません", "GetObject", MB_OK);
		}
		//描画用DC取得
		hdc = ::GetDC(hWnd);
		break;
	case WM_SIZE:
		x = 1;
		break;
	case WM_PAINT:
		PAINTSTRUCT ps;
		hdc = BeginPaint(hWnd, &ps);
		if (DrawIconEx(hdc, 0, 0, icon, windowNS::WINDOW_SUB_WIDTH, windowNS::WINDOW_SUB_HEIGHT, 0, NULL, DI_NORMAL))
		{
			EndPaint(hWnd, &ps);
		}
		break;
	case WM_CLOSE:
		ReleaseDC(NULL, hdc);
		DestroyIcon(icon);
		DestroyWindow(hWnd);
		break;
	case WM_KEYDOWN:case WM_SYSKEYDOWN:
		if (_wParam == VK_ESCAPE)
		{
			PostQuitMessage(0);
		}
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
	// メインウィンドウは更新のみ
	UpdateWindow(hWnd[WINDOWS_ID::MAIN_WINDOWS]);

	// ランチャーの更新・表示
	ShowWindow(hWnd[WINDOWS_ID::SUB_WINDOWS], nCmdShow);
	UpdateWindow(hWnd[WINDOWS_ID::SUB_WINDOWS]);

	//// 画面サイズの取得
	//int x, y;
	//x = GetSystemMetrics(SM_CXSCREEN);
	//y = GetSystemMetrics(SM_CYSCREEN);
	//bool b = SetWindowPos(hWnd[WINDOWS_ID::SUB_WINDOWS],
	//	HWND_TOP, x / 2 - windowNS::WINDOW_SUB_WIDTH / 2, y / 2 - windowNS::WINDOW_SUB_HEIGHT / 2, 
	//	windowNS::WINDOW_SUB_WIDTH, windowNS::WINDOW_SUB_HEIGHT, SWP_SHOWWINDOW);
}

//===================================================================================================================================
// ウインドの再表示
//===================================================================================================================================
void Window::ShowWndAgain(WINDOWS_ID windowID)
{
	// 再表示
	ShowWindow(hWnd[windowID], SW_SHOWDEFAULT);
}

//===================================================================================================================================
// ウインドの終了処理
//===================================================================================================================================
void Window::CloseWindow(WINDOWS_ID windowID)
{
	SendMessage(hWnd[windowID], WM_CLOSE, 0, 0);
}

//===================================================================================================================================
// ウインドハンドルの取得
//===================================================================================================================================
HWND Window::GetWindowHwnd(int windowID)
{
	return hWnd[windowID];
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