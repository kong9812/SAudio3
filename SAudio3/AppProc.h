#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "imguiManager.h"

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class AppProc
{
public:
	AppProc(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext *deviceContext, IDXGISwapChain *swapChain, ID3D11RenderTargetView*renderTargetView);
	~AppProc();

	void Update();
	void Draw(ID3D11RenderTargetView *renderTargetView);

private:
	IDXGISwapChain			*swapChain;			// IDXGISwapChain�\����
	ID3D11Device			*device;			// �f�o�C�X
	ID3D11DeviceContext		*deviceContext;		// �f�o�C�X�R���e�N�X�g
	ID3D11RenderTargetView	*renderTargetView;	// View(�r���[)

	ImGuiManager *imGuiManager;	// ImGui�}�l�[�W���[
};