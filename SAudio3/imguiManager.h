#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"

#include "imGui/imgui.h"
#include "imGui/imgui_impl_win32.h"
#include "imGui/imgui_impl_dx11.h"

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class ImGuiManager
{
public:
	ImGuiManager(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext *deviceContext);
	~ImGuiManager();

	// [ImGui]�V�����t���[���̍쐬
	void CreateNewFrame();

	// [ImGui]�e�X�g
	void test();

private:
	bool showSample;	// [ImGui�t���O]�T���v��UI

};