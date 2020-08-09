#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include <D3D11.h>
#include <D3DX11.h> // see!
#include <D3D11.h>
#include <directxmath.h>
#include <D3Dcompiler.h>
#include <D2D1.h>

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"dxguid.lib")
#pragma comment(lib,"d3dcompiler.lib")

//===================================================================================================================================
// クラス
//===================================================================================================================================
class DirectX11
{
public:
	DirectX11(HWND hWnd);
	~DirectX11();

	// DirectX11の初期化
	bool Init();

	// リサイズ
	void ReSize(LPARAM lParam);

	// デバイスの取得
	ID3D11Device *GetDevice();

	// デバイスコンテクストの取得
	ID3D11DeviceContext *GetDeviceContext();
	
	// IDXGISwapChain構造体の取得
	IDXGISwapChain *GetSwapChain();

	// View(ビュー)構造体の取得
	ID3D11RenderTargetView *GetRenderTargetView();

private:
	DXGI_SWAP_CHAIN_DESC	dxgi;				// DXGI_SWAP_CHAIN_DESC構造体
	IDXGISwapChain			*swapChain;			// IDXGISwapChain構造体
	ID3D11Device			*device;			// デバイス
	ID3D11DeviceContext		*deviceContext;		// デバイスコンテクスト
	D3D_FEATURE_LEVEL		level;				// 機能レベル
	ID3D11RenderTargetView	*renderTargetView;	// View(ビュー)

	// レンダリングターゲットの作成
	void CreateRenderTarget();
};