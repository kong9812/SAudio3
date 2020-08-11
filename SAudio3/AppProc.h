#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "imguiManager.h"
#include "TextureBase.h"

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class AppProc
{
public:
	AppProc(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext *deviceContext, IDXGISwapChain *swapChain, ID3D11RenderTargetView*renderTargetView);
	~AppProc();

	// �X�V����
	void Update(HWND hWnd);
	
	// �`�揈��
	void Draw(ID3D11RenderTargetView *renderTargetView);

	// ���T�C�Y
	void ReSize(bool reSizeFlg);

private:
	IDXGISwapChain			*swapChain;			// IDXGISwapChain�\����
	ID3D11Device			*device;			// �f�o�C�X
	ID3D11DeviceContext		*deviceContext;		// �f�o�C�X�R���e�N�X�g
	ID3D11RenderTargetView	*renderTargetView;	// View(�r���[)

	ImGuiManager			*imGuiManager;		// ImGui�}�l�[�W���[
	TextureBase				*textureBase;		// �e�N�X�`���x�[�X

	bool					reSizeFlg;			// ���C���p�l���̃��T�C�Y�t���O
};