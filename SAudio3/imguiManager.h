#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "TextureBase.h"
#include "SoundBase.h"
#include "XAudio2Manager.h"

//===================================================================================================================================
// �r���h�X�C�b�`
//===================================================================================================================================
// �h�b�L���O�̎g�p��(DONT SWITCH TO FALSE!!!!!)
// true�̂�
#define USE_IMGUI_DOCKING (true)
#if USE_IMGUI_DOCKING
#include "imGui/docking/imgui.h"
#include "imGui/docking/imgui_impl_win32.h"
#include "imGui/docking/imgui_impl_dx11.h"
#else
#include "imGui/imgui.h"
#include "imGui/imgui_impl_win32.h"
#include "imGui/imgui_impl_dx11.h"
#endif

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
namespace imGuiManagerNS
{
	const ImVec2 buttonSize = ImVec2(25, 25);
}

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class ImGuiManager
{
public:
	ImGuiManager(HWND hWnd, ID3D11Device *device, 
		ID3D11DeviceContext *deviceContext, TextureBase *textureBase,
		SoundBase *soundBase);
	~ImGuiManager();

	// [ImGui]�V�����t���[���̍쐬
	void CreateNewFrame();

	// [ImGui]�p�l���̕\��
	void ShowPanel(bool reSize, RECT mainPanelSize);
	void ShowPanel();

private:
	TextureBase *textureBase;	// �e�N�X�`���x�[�X
	SoundBase	*soundBase;		// �T�E���h�x�[�X

	bool showMainPanel;			// [ImGui�t���O]���C���p�l��
	bool showPlayerPanel;		// [ImGui�t���O]�Đ��p�l��
	bool showSoundBasePanel;	// [ImGui�t���O]�T�E���h�x�[�X�p�l��

	bool isPlaying;				// [�v���C���[�p�l��]�Đ���??

	// [ImGui]���T�C�Y
	void ReSize(LONG right, LONG bottom);

	// ���C���p�l��
	void MainPanel();

	// ���j���[�o�[
	void MenuBar();

	// �Đ��p�l��
	void PlayerPanel();

	// �T�E���h�x�[�X�p�l��
	void SoundBasePanel();
};