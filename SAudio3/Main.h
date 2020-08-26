#pragma once
//===================================================================================================================================
// ���C�u�����t���O
//===================================================================================================================================
#define STB_IMAGE_IMPLEMENTATION

//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include <thread>
#include <iostream>
#include <map>
#include <list>
#include <string>
#include <bitset> 
#include <sstream>
#include <cmath>
#include <stdio.h>
#include <crtdbg.h>
#include <windows.h>
#include <Pdh.h>
#include <PdhMsg.h>

#pragma comment(lib, "pdh.lib")
#pragma comment(lib,"winmm.lib")

//===================================================================================================================================
// �x��������
//===================================================================================================================================
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
#define MAIN_APP_NAME				(LPSTR)"SAudio3"
#define SUB_APP_NAME				(LPSTR)"SAudio3_SUB"
#define BYTES_TO_GB(b)			(b) / 1024.0f / 1024.0f / 1024.0f

//===================================================================================================================================
// �G���[���b�Z�[�W�E�R�[�h
//===================================================================================================================================
namespace errorNS
{
	const LPCSTR DXInitError				= "DirectX11 Error(Init)";					// [DirectX11]���������s
	const LPCSTR DXShaderResourceError		= "CreateShaderResourceView Failed";		// [DirectX11]�쐬���s(ID3D11ShaderResourceView)
	
	const LPCSTR ImGuiInitError				= "ImGui Error(Init)";						// [ImGui]���������s
	const LPCSTR ImGuiWin32InitError		= "ImGui Error(Init Win32)";				// [ImGui]���������s(Win32)
	
	const LPCSTR TextureImportError			= "Texture Not Found";						// [Texture]�e�N�X�`�������݂��Ȃ�
	const LPCSTR SoundImportError			= "Sound Not Found";						// [Sound]�T�E���h�����݂��Ȃ�

	const LPCSTR SoundReadError				= "Sound File Can Not Be Open:";			// [Sound]�ǂݍ��ݎ��s
	const LPCSTR SoundResourceError			= "CreateSoundResource Failed";				// [Sound]�쐬���s(SoundResource)

	const LPCSTR XAudio2InitError			= "XAduio2 Error(Init):";					// [XAudio2]���������s
	const LPCSTR XAudio2ComInitError		= "XAduio2 Error(COM Init):";				// [XAudio2]���������s(COM)
	const LPCSTR XAudio2CreateMastering		= "XAduio2 Error(Create Mastering):";		// [XAudio2]�쐬���s(�}�X�^�[�{�C�X)
}