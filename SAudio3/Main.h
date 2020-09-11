#pragma once
//===================================================================================================================================
// ライブラリフラグ
//===================================================================================================================================
#define STB_IMAGE_IMPLEMENTATION

//===================================================================================================================================
// インクルード
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
// 警告無効化
//===================================================================================================================================
#pragma warning(disable:4305)
#pragma warning(disable:4996)
#pragma warning(disable:4018)
#pragma warning(disable:4111)

//===================================================================================================================================
// マクロ定義
//===================================================================================================================================
#define SAFE_RELEASE(p)			if(p){  (p)->Release(); p = NULL; }
#define SAFE_DELETE(p)			if (p){ delete (p);  p = NULL; }
#define SAFE_DELETE_ARRAY(p)    if (p){ delete [] (p);  p = NULL; } 
#define MAIN_APP_NAME				(LPSTR)"SAudio3"
#define SUB_APP_NAME				(LPSTR)"SAudio3_SUB"
#define BYTES_TO_GB(b)			(b) / 1024.0f / 1024.0f / 1024.0f

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