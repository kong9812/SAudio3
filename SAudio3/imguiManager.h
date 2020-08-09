#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"

#include "imGui/imgui.h"
#include "imGui/imgui_impl_win32.h"
#include "imGui/imgui_impl_dx11.h"

//===================================================================================================================================
// クラス
//===================================================================================================================================
class ImGuiManager
{
public:
	ImGuiManager(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext *deviceContext);
	~ImGuiManager();

	// [ImGui]新しいフレームの作成
	void CreateNewFrame();

	// [ImGui]テスト
	void test();

private:
	bool showSample;	// [ImGuiフラグ]サンプルUI

};