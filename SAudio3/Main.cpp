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
	HWND hWnd[WINDOWS_ID::MAX_WINDOWS] =
	{ window->GetWindowHwnd(WINDOWS_ID::MAIN_WINDOWS),
		window->GetWindowHwnd(WINDOWS_ID::SUB_WINDOWS) };
	// DX11�n���h��
	DirectX11 *directX11[WINDOWS_ID::MAX_WINDOWS] = { nullptr };
	
	for (int i = 0; i < WINDOWS_ID::MAX_WINDOWS; i++)
	{
		// DX11�̏�����
		directX11[i] = new DirectX11(hWnd[i]);
		if (!directX11[i]->Init())
		{
			// �G���[���b�Z�[�W
			MessageBox(NULL, errorNS::DXInitError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

			// �����I��
			PostQuitMessage(0);
		}
	}

	// �A�v���v���Z�X�̏�����
	AppProc *appProc = new AppProc(hWnd[WINDOWS_ID::MAIN_WINDOWS],
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetDevice(),
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetDeviceContext(),
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetSwapChain(),
	directX11[WINDOWS_ID::MAIN_WINDOWS]->GetRenderTargetView());

	// �����`���[�̏I������(1�b)
	int launcherEndTime = timeGetTime() + 1000;
	bool isLauncher = true;

	// ���b�Z�[�W���[�v
	while (true)
	{
		// ���T�C�Y
		if (window->GetReSizeFlg() == true)
		{
			if (directX11[WINDOWS_ID::MAIN_WINDOWS]->GetDevice() != NULL && window->GetwParam() != SIZE_MINIMIZED)
			{
				directX11[WINDOWS_ID::MAIN_WINDOWS]->ReSize(window->GetlParam());
				window->SetReSizeFlg(false);
				appProc->ReSize(true);
			}
		}

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
			if (isLauncher)
			{
				if (timeGetTime() >= launcherEndTime)
				{
					window->CloseWindow(WINDOWS_ID::SUB_WINDOWS);
					window->ShowWndAgain(WINDOWS_ID::MAIN_WINDOWS);
					isLauncher = false;
				}
			}
			appProc->Update(hWnd[WINDOWS_ID::MAIN_WINDOWS]);	// �X�V����
			appProc->Draw(directX11[WINDOWS_ID::MAIN_WINDOWS]->GetRenderTargetView());	// �`�揈��
		}
	}

	// �I������
	SAFE_DELETE(appProc)
	for (int i = 0; i < WINDOWS_ID::MAX_WINDOWS; i++)
	{
		SAFE_DELETE(directX11[i])
	}

	return msg.wParam;
}