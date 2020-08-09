#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include <iostream>
#include <stdio.h>
#include <crtdbg.h>
#include <windows.h>
#include <thread>

#pragma comment(lib,"winmm.lib")

#pragma warning(disable:4305)
#pragma warning(disable:4996)
#pragma warning(disable:4018)
#pragma warning(disable:4111)

//===================================================================================================================================
// �}�N����`
//===================================================================================================================================
#define SAFE_RELEASE(p)			if(p){  (p)->Release(); p = NULL; }
#define SAFE_DELETE(p)			if (p){ delete (p);  p = NULL; }
#define SAFE_DELETE_ARRAY(p)    if (p){ delete [] (p);  p = NULL; } 
#define APP_NAME				(LPSTR)"SAudio3"

//===================================================================================================================================
// �G���[���b�Z�[�W�E�R�[�h
//===================================================================================================================================
namespace errorNS
{
	const LPCSTR DXInitError			= "DirectX11 Error(Init)";	// [DirectX11]���������s
	const LPCSTR ImGuiInitError			= "ImGui Error(Init)";		// [ImGui]���������s
	const LPCSTR ImGuiWin32InitError	= "ImGui Error(Init Win32)";// [ImGui]Win32���������s

}