//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "AppProc.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
AppProc::AppProc(HWND hWnd,
	ID3D11Device *_device,
	ID3D11DeviceContext *_deviceContext,
	IDXGISwapChain *_swapChain,
	ID3D11RenderTargetView *_renderTargetView)
{
	// DX11関連
	device				= _device;
	deviceContext		= _deviceContext;
	swapChain			= _swapChain;
	renderTargetView	= _renderTargetView;

	// 初期化
	textureBase		= new TextureBase(device);
	soundBase		= new SoundBase;
	xAudio2Manager	= new XAudio2Manager(soundBase);
	imGuiManager	= new ImGuiManager(hWnd, _device, _deviceContext, textureBase, soundBase, xAudio2Manager);
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
AppProc::~AppProc()
{
	// 終了処理
	SAFE_DELETE(imGuiManager)
	SAFE_DELETE(xAudio2Manager)
	SAFE_DELETE(soundBase)
	SAFE_DELETE(textureBase)
}

//===================================================================================================================================
// 更新処理
//===================================================================================================================================
void AppProc::Update(HWND hWnd)
{
	// [ImGui]新しいフレームの作成
	imGuiManager->CreateNewFrame();
		
	// [ImGui]メインパネル
	if (reSizeFlg)
	{
		RECT rect;
		// ウインドサイズ(描画できる部分)の取得
		if (GetClientRect(hWnd, &rect))
		{
			imGuiManager->ShowPanel(true, rect);
		}
		reSizeFlg = false;
	}
	else
	{
		imGuiManager->ShowPanel();
	}
}

//===================================================================================================================================
// リサイズ
//===================================================================================================================================
void AppProc::ReSize(bool _reSizeFlg)
{
	// ImGuiのリサイズフラグ
	reSizeFlg =_reSizeFlg;
}

//===================================================================================================================================
// 描画処理
//===================================================================================================================================
void AppProc::Draw(ID3D11RenderTargetView *_renderTargetView)
{
	ImGui::Render();
	deviceContext->OMSetRenderTargets(1, &_renderTargetView, NULL);
	deviceContext->ClearRenderTargetView(_renderTargetView, directX11NS::clear);
	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

	swapChain->Present(1,0);
}