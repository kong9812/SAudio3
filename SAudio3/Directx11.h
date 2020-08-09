#pragma once
//===================================================================================================================================
// �C���N���[�h
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
// �N���X
//===================================================================================================================================
class DirectX11
{
public:
	DirectX11(HWND hWnd);
	~DirectX11();

	// DirectX11�̏�����
	bool Init();

	// ���T�C�Y
	void ReSize(LPARAM lParam);

	// �f�o�C�X�̎擾
	ID3D11Device *GetDevice();

	// �f�o�C�X�R���e�N�X�g�̎擾
	ID3D11DeviceContext *GetDeviceContext();
	
	// IDXGISwapChain�\���̂̎擾
	IDXGISwapChain *GetSwapChain();

	// View(�r���[)�\���̂̎擾
	ID3D11RenderTargetView *GetRenderTargetView();

private:
	DXGI_SWAP_CHAIN_DESC	dxgi;				// DXGI_SWAP_CHAIN_DESC�\����
	IDXGISwapChain			*swapChain;			// IDXGISwapChain�\����
	ID3D11Device			*device;			// �f�o�C�X
	ID3D11DeviceContext		*deviceContext;		// �f�o�C�X�R���e�N�X�g
	D3D_FEATURE_LEVEL		level;				// �@�\���x��
	ID3D11RenderTargetView	*renderTargetView;	// View(�r���[)

	// �����_�����O�^�[�Q�b�g�̍쐬
	void CreateRenderTarget();
};