//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Window.h"
#include "imguiManager.h"

//===================================================================================================================================
// ���[�J���ϐ�
//===================================================================================================================================
Window* window = NULL;	// �E�C���h�E�v���V�[�W���p

//===================================================================================================================================
// �E�C���h�E�v���V�[�W��(���ڂɎg���Ȃ�����)
//===================================================================================================================================
LRESULT CALLBACK wndProc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK wndProc(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	return window->WndProc(wnd, msg, wparam, lparam);
}

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
Window::Window()
{
	// ���������[�N�̌��o
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	// �X���b�h���s��
	unsigned int count = std::thread::hardware_concurrency();

	// ���T�C�Y�t���O�̏�����
	reSizeFlg = false;
}

//===================================================================================================================================
// ������
//===================================================================================================================================
HRESULT Window::Init(HINSTANCE hInstance)
{
	// �v���V�[�W���p
	window = this;

	// �E�B���h�E�N���X�̓o�^
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = (WNDPROC)wndProc;  //  �E�B���h�E�v���V�[�W���̓o�^
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;  //  �A�v���P�[�V�����C���X�^���X
	wcex.hIcon = LoadIcon(NULL, MAKEINTRESOURCE(IDI_APPLICATION));
	wcex.hCursor = LoadCursor(NULL, MAKEINTRESOURCE(IDC_ARROW));
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = NULL;
	wcex.lpszClassName = APP_NAME;  //  �E�B���h�E�N���X��
	wcex.hIconSm = LoadIcon(NULL, IDI_APPLICATION);
	RegisterClassEx(&wcex);

	//  �E�B���h�E�̐���
	hWnd = CreateWindowEx(
		WS_EX_OVERLAPPEDWINDOW,
		wcex.lpszClassName,
		APP_NAME,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		0,
		852,
		480,
		NULL,
		NULL,
		hInstance,
		NULL);

	if (!hWnd)
		return FALSE;
}

//===================================================================================================================================
// �E�C���h�E�v���V�[�W��
//===================================================================================================================================
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK Window::WndProc(HWND hWnd, UINT message, WPARAM _wParam, LPARAM _lParam)
{
	if (ImGui_ImplWin32_WndProcHandler(hWnd, message, _wParam, _lParam))
		return true;

	switch (message)
	{
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_KEYDOWN:case WM_SYSKEYDOWN:
		if (_wParam == VK_ESCAPE)
		{
			PostQuitMessage(0);
		}
		break;
	case WM_SIZE:
		reSizeFlg = true;
		wParam = _wParam;
		lParam = _lParam;
		break;
	default:
		return DefWindowProc(hWnd, message, _wParam, _lParam);
	}

	return DefWindowProc(hWnd, message, _wParam, _lParam);
}

//===================================================================================================================================
// �E�C���h�̕\��
//===================================================================================================================================
void Window::ShowWnd(int nCmdShow)
{
#if 0
	// �E�B���h�E�T�C�Y�̒���
	GetWindowRect(hWnd, &bounds);
	GetClientRect(hWnd, &client);
	MoveWindow(hWnd, bounds.left, bounds.top,
		852 * 2 - client.right,
		480 * 2 - client.bottom,
		false);
#endif

	// �E�B���h�E�̍ĕ\��
	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);
}

//===================================================================================================================================
// �E�C���h�n���h���̎擾
//===================================================================================================================================
HWND Window::GetWindowHwnd()
{
	return hWnd;
}

//===================================================================================================================================
// ���T�C�Y�t���O�̎擾
//===================================================================================================================================
bool Window::GetReSizeFlg()
{
	return reSizeFlg;
}
//===================================================================================================================================
// ���T�C�Y�t���O�̐ݒ�
//===================================================================================================================================
void Window::SetReSizeFlg(bool _reSizeFlg)
{
	reSizeFlg = _reSizeFlg;
}

//===================================================================================================================================
// ���T�C�Y�f�[�^�̎擾(wParam)
//===================================================================================================================================
WPARAM Window::GetwParam()
{
	return wParam;
}

//===================================================================================================================================
// ���T�C�Y�f�[�^�̎擾(lParam)
//===================================================================================================================================
LPARAM Window::GetlParam()
{
	return lParam;
}