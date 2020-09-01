#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "icon.h"

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
namespace windowNS
{
	const int WINDOW_MAIN_WIDTH	= 1920;
	const int WINDOW_MAIN_HEIGHT = 1080;
	const int WINDOW_SUB_WIDTH = 256;
	const int WINDOW_SUB_HEIGHT = 256;
}
enum WINDOWS_ID
{
	MAIN_WINDOWS,
	SUB_WINDOWS,
	MAX_WINDOWS
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class Window
{
public:
	Window();
	~Window() {};

	// �E�C���h�E�v���V�[�W��
	LRESULT WndProc1(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	LRESULT WndProc2(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	// �E�C���h�̏�����
	HRESULT Init(HINSTANCE hInstance);

	// �E�C���h�̕\��
	void ShowWnd(int nCmdShow);
	// �E�C���h�̍ĕ\��
	void ShowWndAgain(WINDOWS_ID windowID);

	// �E�C���h�̏I������
	void CloseWindow(WINDOWS_ID windowID);

	// �E�C���h�n���h���̎擾
	HWND GetWindowHwnd(int windowID);

	// ���T�C�Y�t���O�̎擾
	bool GetReSizeFlg();
	// ���T�C�Y�t���O�̐ݒ�
	void SetReSizeFlg(bool _reSizeFlg);

	// ���T�C�Y�f�[�^�̎擾
	WPARAM GetwParam();
	LPARAM GetlParam();

private:
	WNDCLASSEX wcex[WINDOWS_ID::MAX_WINDOWS];	// �E�B���h�E�N���X�\����
	HWND hWnd[WINDOWS_ID::MAX_WINDOWS];			// �E�B���h�E�n���h��
	RECT bounds, client;	// RECT�\����
	bool reSizeFlg;			// ���T�C�Y�t���O
	WPARAM wParam;			// ���T�C�Y�p
	LPARAM lParam;			// ���T�C�Y�p
	HICON icon;				// �����`���[�摜
	HDC	hdc;				// �����`���[�`��p
};