#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include <thread>
#include <iostream>
#include <map>
#include <string>
#include <bitset> 
#include <sstream>
#include <stdio.h>
#include <crtdbg.h>
#include <windows.h>

#pragma comment(lib,"winmm.lib")

//===================================================================================================================================
// 警告無効化
//===================================================================================================================================
#pragma warning(disable:4305)
#pragma warning(disable:4996)
#pragma warning(disable:4018)
#pragma warning(disable:4111)

//===================================================================================================================================
// ライブラリフラグ
//===================================================================================================================================
#define STB_IMAGE_IMPLEMENTATION

//===================================================================================================================================
// マクロ定義
//===================================================================================================================================
#define SAFE_RELEASE(p)			if(p){  (p)->Release(); p = NULL; }
#define SAFE_DELETE(p)			if (p){ delete (p);  p = NULL; }
#define SAFE_DELETE_ARRAY(p)    if (p){ delete [] (p);  p = NULL; } 
#define APP_NAME				(LPSTR)"SAudio3"

//===================================================================================================================================
// エラーメッセージ・コード
//===================================================================================================================================
namespace errorNS
{
	const LPCSTR DXInitError				= "DirectX11 Error(Init)";					// [DirectX11]初期化失敗
	const LPCSTR DXShaderResourceError		= "CreateShaderResourceView Failed";		// [DirectX11]作成失敗(ID3D11ShaderResourceView)
	
	const LPCSTR ImGuiInitError				= "ImGui Error(Init)";						// [ImGui]初期化失敗
	const LPCSTR ImGuiWin32InitError		= "ImGui Error(Init Win32)";				// [ImGui]初期化失敗(Win32)
	
	const LPCSTR TextureImportError			= "Texture Not Found";						// [Texture]テクスチャが存在しない
	const LPCSTR SoundImportError			= "Sound Not Found";						// [Sound]サウンドが存在しない

	const LPCSTR SoundReadError				= "Sound File Can Not Be Open:";			// [Sound]読み込み失敗
	const LPCSTR SoundResourceError			= "CreateSoundResource Failed";				// [Sound]作成失敗(SoundResource)

	const LPCSTR XAudio2InitError			= "XAduio2 Error(Init):";					// [XAudio2]初期化失敗
	const LPCSTR XAudio2ComInitError		= "XAduio2 Error(COM Init):";				// [XAudio2]初期化失敗(COM)
	const LPCSTR XAudio2CreateMastering		= "XAduio2 Error(Create Mastering):";		// [XAudio2]作成失敗(マスターボイス)
}