#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "imguiManager.h"

//===================================================================================================================================
// クラス
//===================================================================================================================================
class AppProc
{
public:
	AppProc(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext *deviceContext, IDXGISwapChain *swapChain, ID3D11RenderTargetView*renderTargetView);
	~AppProc();

	void Update();
	void Draw(ID3D11RenderTargetView *renderTargetView);

private:
	IDXGISwapChain			*swapChain;			// IDXGISwapChain構造体
	ID3D11Device			*device;			// デバイス
	ID3D11DeviceContext		*deviceContext;		// デバイスコンテクスト
	ID3D11RenderTargetView	*renderTargetView;	// View(ビュー)

	ImGuiManager *imGuiManager;	// ImGuiマネージャー
};