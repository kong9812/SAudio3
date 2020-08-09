#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"

//===================================================================================================================================
// �}�N����`
//===================================================================================================================================
namespace windowNS
{
	const int WINDOW_WIDTH = 1920;
	const int WINDOW_HEIGHT = 1080;
}

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class Window
{
public:
	Window();
	~Window() {};

	// �E�C���h�E�v���V�[�W��
	LRESULT WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	// �E�C���h�̏�����
	HRESULT Init(HINSTANCE hInstance);

	// �E�C���h�̕\��
	void ShowWnd(int nCmdShow);

	// �E�C���h�n���h���̎擾
	HWND GetWindowHwnd();

	// ���T�C�Y�t���O�̎擾
	bool GetReSizeFlg();
	// ���T�C�Y�t���O�̐ݒ�
	void SetReSizeFlg(bool _reSizeFlg);

	// ���T�C�Y�f�[�^�̎擾
	WPARAM GetwParam();
	LPARAM GetlParam();

private:
	WNDCLASSEX wcex;		// �E�B���h�E�N���X�\����
	HWND hWnd;				// �E�B���h�E�n���h��
	RECT bounds, client;	// RECT�\����
	bool reSizeFlg;			// ���T�C�Y�t���O
	WPARAM wParam;			// ���T�C�Y�p
	LPARAM lParam;			// ���T�C�Y�p
};