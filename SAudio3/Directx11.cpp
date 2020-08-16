//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Directx11.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
DirectX11::DirectX11(HWND hWnd)
{
	// ウインド
	HWND window_handle = hWnd;	// ウィンドウハンドル
	RECT rect;					// RECT 構造体
	GetClientRect(window_handle, &rect);	// ウィンドのサイズ

	// DXGI_SWAP_CHAIN_DESC構造体の初期化
	ZeroMemory(&dxgi, sizeof(DXGI_SWAP_CHAIN_DESC));
	dxgi.BufferCount = 1;								// バッファの数
	dxgi.BufferDesc.Width = (rect.right - rect.left);	// バッファの横幅
	dxgi.BufferDesc.Height = (rect.bottom - rect.top);	// バッファの縦幅
	dxgi.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;// カラーフォーマット
	dxgi.BufferDesc.RefreshRate.Numerator = 60;			// リフレッシュレートの分母
	dxgi.BufferDesc.RefreshRate.Denominator = 1;		// リフレッシュレートの分子
	dxgi.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;	// バッファの使い方 Usage => 使用方法
	dxgi.OutputWindow = window_handle;					// 出力対象のウィンドウハンドル
	dxgi.SampleDesc.Count = 1;							// マルチサンプリングのサンプル数(未使用は1)
	dxgi.SampleDesc.Quality = 0;						// マルチサンプリングの品質(未使用は0)
	dxgi.Windowed = true;								// ウィンドウモード指定

	// View(ビュー)の初期化
	renderTargetView = NULL;
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
DirectX11::~DirectX11()
{
	// DirectX11の終了処理
	SAFE_RELEASE(renderTargetView)
	SAFE_RELEASE(deviceContext)
	SAFE_RELEASE(device)
	SAFE_RELEASE(swapChain)
}

//===================================================================================================================================
// DirectX11の初期化
//===================================================================================================================================
bool DirectX11::Init()
{
	if (FAILED(D3D11CreateDeviceAndSwapChain(
		nullptr,			// ビデオアダプタ指定(nullptrは既定)
		D3D_DRIVER_TYPE_HARDWARE,	// ドライバのタイプ
		nullptr,			// D3D_DRIVER_TYPE_SOFTWARE指定時に使用
		0,					// フラグ指定
		nullptr,			// D3D_FEATURE_LEVEL指定で自分で定義した配列を指定可能
		0,					// 上のD3D_FEATURE_LEVEL配列の要素数
		D3D11_SDK_VERSION,	// SDKバージョン
		&dxgi,				// DXGI_SWAP_CHAIN_DESC
		&swapChain,			// 関数成功時のSwapChainの出力先 
		&device,			// 関数成功時のDeviceの出力先
		&level,				// 成功したD3D_FEATURE_LEVELの出力先
		&deviceContext)))	// 関数成功時のContextの出力先
	{
		return false;
	}

	// レンダリングターゲットの作成
	CreateRenderTarget();

	return true;
}

//===================================================================================================================================
// リサイズ
//===================================================================================================================================
void DirectX11::ReSize(LPARAM lParam)
{
	// 古いレンダリングターゲットの削除
	SAFE_RELEASE(renderTargetView)

	// リサイズ
	swapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
	
	// レンダリングターゲットの作成
	CreateRenderTarget();
}

//===================================================================================================================================
// レンダリングターゲットの作成
//===================================================================================================================================
void DirectX11::CreateRenderTarget()
{
	ID3D11Texture2D* backBuffer;
	swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
	device->CreateRenderTargetView(backBuffer, NULL, &renderTargetView);
	SAFE_RELEASE(backBuffer)
}

//===================================================================================================================================
// デバイスの取得
//===================================================================================================================================
ID3D11Device *DirectX11::GetDevice()
{
	return device;
}

//===================================================================================================================================
// デバイスコンテクストの取得
//===================================================================================================================================
ID3D11DeviceContext *DirectX11::GetDeviceContext()
{
	return deviceContext;
}

//===================================================================================================================================
// IDXGISwapChain構造体の取得
//===================================================================================================================================
IDXGISwapChain *DirectX11::GetSwapChain()
{
	return swapChain;
}

//===================================================================================================================================
// View(ビュー)構造体の取得
//===================================================================================================================================
ID3D11RenderTargetView *DirectX11::GetRenderTargetView()
{
	return renderTargetView;
}