#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "imguiManager.h"
#include "TextureBase.h"

//===================================================================================================================================
// クラス
//===================================================================================================================================
class AppProc
{
public:
	AppProc(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext *deviceContext, IDXGISwapChain *swapChain, ID3D11RenderTargetView*renderTargetView);
	~AppProc();

	// 更新処理
	void Update(HWND hWnd);
	
	// 描画処理
	void Draw(ID3D11RenderTargetView *renderTargetView);

	// リサイズ
	void ReSize(bool reSizeFlg);

private:
	IDXGISwapChain			*swapChain;			// IDXGISwapChain構造体
	ID3D11Device			*device;			// デバイス
	ID3D11DeviceContext		*deviceContext;		// デバイスコンテクスト
	ID3D11RenderTargetView	*renderTargetView;	// View(ビュー)

	ImGuiManager			*imGuiManager;		// ImGuiマネージャー
	TextureBase				*textureBase;		// テクスチャベース

	bool					reSizeFlg;			// メインパネルのリサイズフラグ
};