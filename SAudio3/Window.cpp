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
LRESULT CALLBACK mainWndProc(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK mainWndProc(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	return window->WndProc1(wnd, msg, wparam, lparam);
}
LRESULT CALLBACK subWndProc2(HWND wnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK subWndProc2(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	return window->WndProc2(wnd, msg, wparam, lparam);
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
	wcex[WINDOWS_ID::MAIN_WINDOWS].cbSize = sizeof(WNDCLASSEX);
	wcex[WINDOWS_ID::MAIN_WINDOWS].style = CS_HREDRAW | CS_VREDRAW;
	wcex[WINDOWS_ID::MAIN_WINDOWS].lpfnWndProc = (WNDPROC)mainWndProc;  //  �E�B���h�E�v���V�[�W���̓o�^
	wcex[WINDOWS_ID::MAIN_WINDOWS].cbClsExtra = 0;
	wcex[WINDOWS_ID::MAIN_WINDOWS].cbWndExtra = 0;
	wcex[WINDOWS_ID::MAIN_WINDOWS].hInstance = hInstance;  //  �A�v���P�[�V�����C���X�^���X
	wcex[WINDOWS_ID::MAIN_WINDOWS].hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
	wcex[WINDOWS_ID::MAIN_WINDOWS].hCursor = LoadCursor(NULL, MAKEINTRESOURCE(IDC_ARROW));
	wcex[WINDOWS_ID::MAIN_WINDOWS].hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex[WINDOWS_ID::MAIN_WINDOWS].lpszMenuName = NULL;
	wcex[WINDOWS_ID::MAIN_WINDOWS].lpszClassName = MAIN_APP_NAME;  //  �E�B���h�E�N���X��
	wcex[WINDOWS_ID::MAIN_WINDOWS].hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
	RegisterClassEx(&wcex[WINDOWS_ID::MAIN_WINDOWS]);

	wcex[WINDOWS_ID::SUB_WINDOWS].cbSize = sizeof(WNDCLASSEX);
	wcex[WINDOWS_ID::SUB_WINDOWS].style = CS_HREDRAW | CS_VREDRAW;
	wcex[WINDOWS_ID::SUB_WINDOWS].lpfnWndProc = (WNDPROC)subWndProc2;  //  �E�B���h�E�v���V�[�W���̓o�^
	wcex[WINDOWS_ID::SUB_WINDOWS].cbClsExtra = 0;
	wcex[WINDOWS_ID::SUB_WINDOWS].cbWndExtra = 0;
	wcex[WINDOWS_ID::SUB_WINDOWS].hInstance = hInstance;  //  �A�v���P�[�V�����C���X�^���X
	wcex[WINDOWS_ID::SUB_WINDOWS].hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(NULL));
	wcex[WINDOWS_ID::SUB_WINDOWS].hCursor = LoadCursor(NULL, MAKEINTRESOURCE(NULL));
	wcex[WINDOWS_ID::SUB_WINDOWS].hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex[WINDOWS_ID::SUB_WINDOWS].lpszMenuName = NULL;
	wcex[WINDOWS_ID::SUB_WINDOWS].lpszClassName = SUB_APP_NAME;  //  �E�B���h�E�N���X��
	wcex[WINDOWS_ID::SUB_WINDOWS].hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(NULL));
	RegisterClassEx(&wcex[WINDOWS_ID::SUB_WINDOWS]);

	//  �E�B���h�E�̐���
	hWnd[WINDOWS_ID::MAIN_WINDOWS] = CreateWindowEx(
		WS_EX_OVERLAPPEDWINDOW,
		wcex[WINDOWS_ID::MAIN_WINDOWS].lpszClassName,
		MAIN_APP_NAME,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		0,
		windowNS::WINDOW_WIDTH,
		windowNS::WINDOW_HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL);
	hWnd[WINDOWS_ID::SUB_WINDOWS] = CreateWindowEx(
		WS_EX_OVERLAPPEDWINDOW,
		wcex[WINDOWS_ID::SUB_WINDOWS].lpszClassName,
		MAIN_APP_NAME,
		WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT,
		0,
		windowNS::WINDOW_WIDTH,
		windowNS::WINDOW_HEIGHT,
		hWnd[WINDOWS_ID::MAIN_WINDOWS],
		NULL,
		hInstance,
		NULL);

	if (!hWnd[WINDOWS_ID::MAIN_WINDOWS] || !hWnd[WINDOWS_ID::SUB_WINDOWS])
		return FALSE;
}

//===================================================================================================================================
// �E�C���h�E�v���V�[�W��
//===================================================================================================================================
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK Window::WndProc1(HWND hWnd, UINT message, WPARAM _wParam, LPARAM _lParam)
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
		// ���T�C�Y�f�[�^
		reSizeFlg = true;
		wParam = _wParam;
		lParam = _lParam;
		break;
	default:
		return DefWindowProc(hWnd, message, _wParam, _lParam);
	}

	return DefWindowProc(hWnd, message, _wParam, _lParam);
}
LRESULT CALLBACK Window::WndProc2(HWND hWnd, UINT message, WPARAM _wParam, LPARAM _lParam)
{
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
	for (int i = 0; i < WINDOWS_ID::MAX_WINDOWS; i++)
	{
		ShowWindow(hWnd[i], nCmdShow);
		UpdateWindow(hWnd[i]);
	}
}

//===================================================================================================================================
// �E�C���h�n���h���̎擾
//===================================================================================================================================
HWND Window::GetWindowHwnd(int windowID)
{
	return hWnd[windowID];
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