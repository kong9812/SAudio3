//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "AppProc.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
AppProc::AppProc(HWND hWnd,
	ID3D11Device *_device,
	ID3D11DeviceContext *_deviceContext,
	IDXGISwapChain *_swapChain,
	ID3D11RenderTargetView *_renderTargetView)
{
	// DX11�֘A
	device				= _device;
	deviceContext		= _deviceContext;
	swapChain			= _swapChain;
	renderTargetView	= _renderTargetView;

	// ������
	textureBase		= new TextureBase(device);
	soundBase		= new SoundBase;
	xAudio2Manager	= new XAudio2Manager(soundBase);
	imGuiManager	= new ImGuiManager(hWnd, _device, _deviceContext, textureBase, soundBase, xAudio2Manager);
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
AppProc::~AppProc()
{
	// �I������
	SAFE_DELETE(imGuiManager)
	SAFE_DELETE(xAudio2Manager)
	SAFE_DELETE(soundBase)
	SAFE_DELETE(textureBase)
}

//===================================================================================================================================
// �X�V����
//===================================================================================================================================
void AppProc::Update(HWND hWnd)
{
	// [ImGui]�V�����t���[���̍쐬
	imGuiManager->CreateNewFrame();
		
	// [ImGui]���C���p�l��
	if (reSizeFlg)
	{
		RECT rect;
		// �E�C���h�T�C�Y(�`��ł��镔��)�̎擾
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
// ���T�C�Y
//===================================================================================================================================
void AppProc::ReSize(bool _reSizeFlg)
{
	// ImGui�̃��T�C�Y�t���O
	reSizeFlg =_reSizeFlg;
}

//===================================================================================================================================
// �`�揈��
//===================================================================================================================================
void AppProc::Draw(ID3D11RenderTargetView *_renderTargetView)
{
	ImGui::Render();
	deviceContext->OMSetRenderTargets(1, &_renderTargetView, NULL);
	deviceContext->ClearRenderTargetView(_renderTargetView, directX11NS::clear);
	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

	swapChain->Present(1,0);
}