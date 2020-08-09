//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Window.h"
#include "imguiManager.h"
#include "AppProc.h"

//===================================================================================================================================
// �v���g�^�C�v�錾
//===================================================================================================================================
WPARAM MessageLoop(Window *window, MSG msg);	// ���b�Z�[�W���[�v

//===================================================================================================================================
// WinMain
//===================================================================================================================================
INT APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	// �E�C���h
	Window *window = new Window;

	// �E�C���h�̏�����
	if (!window->Init(hInstance))
		return FALSE;

	// �E�B���h�E�̍ĕ\��
	window->ShowWnd(nCmdShow);

	// ���b�Z�[�W���[�v&�I������
	MSG msg = { 0 };
	ZeroMemory(&msg, sizeof(msg));
	return (int)MessageLoop(window, msg);
}

//===================================================================================================================================
// ���b�Z�[�W���[�v
//===================================================================================================================================
WPARAM MessageLoop(Window *window, MSG msg)
{
	// �E�C���h�n���h��
	HWND hWnd = window->GetWindowHwnd();

	// DX11�̏�����
	DirectX11 *directX11 = new DirectX11(hWnd);
	if (!directX11->Init())
	{
		// �G���[���b�Z�[�W
		MessageBox(NULL, errorNS::DXInitError, APP_NAME, (MB_OK | MB_ICONERROR));

		// �����I��
		PostQuitMessage(0);
	}

	// �A�v���v���Z�X�̏�����
	AppProc *appProc = new AppProc(hWnd,
		directX11->GetDevice(),
		directX11->GetDeviceContext(),
		directX11->GetSwapChain(),
		directX11->GetRenderTargetView());

	// ���b�Z�[�W���[�v
	while (true)
	{
		// ���b�Z�[�W���
		if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT)
			{
				break;
			}
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else
		{
			appProc->Update();	// �X�V����
			appProc->Draw(directX11->GetRenderTargetView());	// �`�揈��
		}

		// ���T�C�Y
		if (window->GetReSizeFlg() == true)
		{
			if (directX11->GetDevice() != NULL && window->GetwParam() != SIZE_MINIMIZED)
			{
				directX11->ReSize(window->GetlParam());
				window->SetReSizeFlg(false);
			}
		}
	}

	// �I������
	SAFE_DELETE(appProc)
	SAFE_DELETE(directX11)

	return msg.wParam;
}