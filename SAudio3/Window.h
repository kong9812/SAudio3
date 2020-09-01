#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "icon.h"

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
namespace windowNS
{
	const int WINDOW_MAIN_WIDTH	= 1920;
	const int WINDOW_MAIN_HEIGHT = 1080;
	const int WINDOW_SUB_WIDTH = 256;
	const int WINDOW_SUB_HEIGHT = 256;
}
enum WINDOWS_ID
{
	MAIN_WINDOWS,
	SUB_WINDOWS,
	MAX_WINDOWS
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class Window
{
public:
	Window();
	~Window() {};

	// ウインドウプロシージャ
	LRESULT WndProc1(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	LRESULT WndProc2(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	// ウインドの初期化
	HRESULT Init(HINSTANCE hInstance);

	// ウインドの表示
	void ShowWnd(int nCmdShow);
	// ウインドの再表示
	void ShowWndAgain(WINDOWS_ID windowID);

	// ウインドの終了処理
	void CloseWindow(WINDOWS_ID windowID);

	// ウインドハンドルの取得
	HWND GetWindowHwnd(int windowID);

	// リサイズフラグの取得
	bool GetReSizeFlg();
	// リサイズフラグの設定
	void SetReSizeFlg(bool _reSizeFlg);

	// リサイズデータの取得
	WPARAM GetwParam();
	LPARAM GetlParam();

private:
	WNDCLASSEX wcex[WINDOWS_ID::MAX_WINDOWS];	// ウィンドウクラス構造体
	HWND hWnd[WINDOWS_ID::MAX_WINDOWS];			// ウィンドウハンドル
	RECT bounds, client;	// RECT構造体
	bool reSizeFlg;			// リサイズフラグ
	WPARAM wParam;			// リサイズ用
	LPARAM lParam;			// リサイズ用
	HICON icon;				// ランチャー画像
	HDC	hdc;				// ランチャー描画用
};