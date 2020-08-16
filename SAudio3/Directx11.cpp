//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Directx11.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
DirectX11::DirectX11(HWND hWnd)
{
	// �E�C���h
	HWND window_handle = hWnd;	// �E�B���h�E�n���h��
	RECT rect;					// RECT �\����
	GetClientRect(window_handle, &rect);	// �E�B���h�̃T�C�Y

	// DXGI_SWAP_CHAIN_DESC�\���̂̏�����
	ZeroMemory(&dxgi, sizeof(DXGI_SWAP_CHAIN_DESC));
	dxgi.BufferCount = 1;								// �o�b�t�@�̐�
	dxgi.BufferDesc.Width = (rect.right - rect.left);	// �o�b�t�@�̉���
	dxgi.BufferDesc.Height = (rect.bottom - rect.top);	// �o�b�t�@�̏c��
	dxgi.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;// �J���[�t�H�[�}�b�g
	dxgi.BufferDesc.RefreshRate.Numerator = 60;			// ���t���b�V�����[�g�̕���
	dxgi.BufferDesc.RefreshRate.Denominator = 1;		// ���t���b�V�����[�g�̕��q
	dxgi.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;	// �o�b�t�@�̎g���� Usage => �g�p���@
	dxgi.OutputWindow = window_handle;					// �o�͑Ώۂ̃E�B���h�E�n���h��
	dxgi.SampleDesc.Count = 1;							// �}���`�T���v�����O�̃T���v����(���g�p��1)
	dxgi.SampleDesc.Quality = 0;						// �}���`�T���v�����O�̕i��(���g�p��0)
	dxgi.Windowed = true;								// �E�B���h�E���[�h�w��

	// View(�r���[)�̏�����
	renderTargetView = NULL;
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
DirectX11::~DirectX11()
{
	// DirectX11�̏I������
	SAFE_RELEASE(renderTargetView)
	SAFE_RELEASE(deviceContext)
	SAFE_RELEASE(device)
	SAFE_RELEASE(swapChain)
}

//===================================================================================================================================
// DirectX11�̏�����
//===================================================================================================================================
bool DirectX11::Init()
{
	if (FAILED(D3D11CreateDeviceAndSwapChain(
		nullptr,			// �r�f�I�A�_�v�^�w��(nullptr�͊���)
		D3D_DRIVER_TYPE_HARDWARE,	// �h���C�o�̃^�C�v
		nullptr,			// D3D_DRIVER_TYPE_SOFTWARE�w�莞�Ɏg�p
		0,					// �t���O�w��
		nullptr,			// D3D_FEATURE_LEVEL�w��Ŏ����Œ�`�����z����w��\
		0,					// ���D3D_FEATURE_LEVEL�z��̗v�f��
		D3D11_SDK_VERSION,	// SDK�o�[�W����
		&dxgi,				// DXGI_SWAP_CHAIN_DESC
		&swapChain,			// �֐���������SwapChain�̏o�͐� 
		&device,			// �֐���������Device�̏o�͐�
		&level,				// ��������D3D_FEATURE_LEVEL�̏o�͐�
		&deviceContext)))	// �֐���������Context�̏o�͐�
	{
		return false;
	}

	// �����_�����O�^�[�Q�b�g�̍쐬
	CreateRenderTarget();

	return true;
}

//===================================================================================================================================
// ���T�C�Y
//===================================================================================================================================
void DirectX11::ReSize(LPARAM lParam)
{
	// �Â������_�����O�^�[�Q�b�g�̍폜
	SAFE_RELEASE(renderTargetView)

	// ���T�C�Y
	swapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
	
	// �����_�����O�^�[�Q�b�g�̍쐬
	CreateRenderTarget();
}

//===================================================================================================================================
// �����_�����O�^�[�Q�b�g�̍쐬
//===================================================================================================================================
void DirectX11::CreateRenderTarget()
{
	ID3D11Texture2D* backBuffer;
	swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
	device->CreateRenderTargetView(backBuffer, NULL, &renderTargetView);
	SAFE_RELEASE(backBuffer)
}

//===================================================================================================================================
// �f�o�C�X�̎擾
//===================================================================================================================================
ID3D11Device *DirectX11::GetDevice()
{
	return device;
}

//===================================================================================================================================
// �f�o�C�X�R���e�N�X�g�̎擾
//===================================================================================================================================
ID3D11DeviceContext *DirectX11::GetDeviceContext()
{
	return deviceContext;
}

//===================================================================================================================================
// IDXGISwapChain�\���̂̎擾
//===================================================================================================================================
IDXGISwapChain *DirectX11::GetSwapChain()
{
	return swapChain;
}

//===================================================================================================================================
// View(�r���[)�\���̂̎擾
//===================================================================================================================================
ID3D11RenderTargetView *DirectX11::GetRenderTargetView()
{
	return renderTargetView;
}