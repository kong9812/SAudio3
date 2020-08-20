//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
XAudio2Manager::XAudio2Manager(SoundBase *_soundBase)
{
	HRESULT hr = E_FAIL;

	// COMの初期化
	hr = (CoInitializeEx(nullptr, COINIT_MULTITHREADED));
	if (hr != S_OK)
	{
		// エラーメッセージ
		std::stringstream tmpMsg("");
		tmpMsg << "0x" << std::hex << hr;
		std::string errorMsg = errorNS::XAudio2ComInitError + tmpMsg.str();
		MessageBox(NULL, errorMsg.c_str(), APP_NAME, (MB_OK | MB_ICONERROR));
		tmpMsg.clear();
		errorMsg.clear();

		// 強制終了
		PostQuitMessage(0);
	}

	// XAudio2の初期化
#if ( _WIN32_WINNT < _WIN32_WINNT_WIN8) && defined(_DEBUG)
	// Win7のデバッグ機能
	hr = XAudio2Create(&XAudio2, XAUDIO2_DEBUG_ENGINE, XAUDIO2_DEFAULT_PROCESSOR);
#else
	hr = XAudio2Create(&XAudio2, NULL, XAUDIO2_DEFAULT_PROCESSOR);
#endif
	if (hr != S_OK)
	{
		// エラーメッセージ
		std::stringstream tmpMsg("");
		tmpMsg << "0x" << std::hex << hr;
		std::string errorMsg = errorNS::XAudio2InitError + tmpMsg.str();
		MessageBox(NULL, errorMsg.c_str(), APP_NAME, (MB_OK | MB_ICONERROR));
		tmpMsg.clear();
		errorMsg.clear();

		// 強制終了
		PostQuitMessage(0);
	}

	//std::string teststring = "TEST";
	//std::stringstream hValStr;
	//for (std::size_t i = 0; i < teststring.length(); i++)
	//{
	//	int hValInt = (char)teststring[i];
	//	hValStr << "0x" << std::hex << hValInt << " ";
	//}

#if ( _WIN32_WINNT > _WIN32_WINNT_WIN7) && defined(_DEBUG)
	// Win7以降のデバッグ機能
	XAUDIO2_DEBUG_CONFIGURATION debugConfig{ NULL };
	debugConfig.LogFileline = true;
	debugConfig.LogFunctionName = true;
	debugConfig.LogTiming = true;
	debugConfig.LogThreadID = true;
	debugConfig.TraceMask = XAUDIO2_LOG_ERRORS | XAUDIO2_LOG_WARNINGS |
		XAUDIO2_LOG_INFO | XAUDIO2_LOG_DETAIL |
		XAUDIO2_LOG_API_CALLS | XAUDIO2_LOG_FUNC_CALLS |
		XAUDIO2_LOG_TIMING | XAUDIO2_LOG_LOCKS |
		XAUDIO2_LOG_MEMORY | XAUDIO2_LOG_STREAMING;
	debugConfig.BreakMask = XAUDIO2_LOG_ERRORS | XAUDIO2_LOG_WARNINGS; 
	XAudio2->SetDebugConfiguration(&debugConfig);
#endif

	// マスターボイスの作成
	XAudio2MasteringVoice = CreateMasterVoice(XAudio2);
	if (XAudio2MasteringVoice == nullptr)
	{
		// 強制終了
		PostQuitMessage(0);
	}

	soundBase = _soundBase;
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
XAudio2Manager::~XAudio2Manager()
{
	// 連想配列の削除
	if (voiceResource.size() > NULL)
	{
		// サウンドリソースの終了処理
		auto begin = voiceResource.begin();
		auto end = voiceResource.end();
		for (auto i = begin; i != end; i++)
		{
			// ソースボイスの終了処理
			//i->second.sourceVoice->Stop();
			SAFE_DESTROY_VOICE(i->second.sourceVoice)
		}
		voiceResource.clear();
	}

	// マスターボイスの終了処理
	SAFE_DESTROY_VOICE(XAudio2MasteringVoice)

	// XAudio2の終了処理
	SAFE_RELEASE(XAudio2)
	
	// COMの終了処理
	CoUninitialize();
}

//===================================================================================================================================
// マスターボイスの作成
//===================================================================================================================================
IXAudio2MasteringVoice *XAudio2Manager::CreateMasterVoice(IXAudio2 *xAudio2)
{
	HRESULT hr = E_FAIL;
	IXAudio2MasteringVoice *tmpXAudio2MasteringVoice = nullptr;

	// マスターボイスの作成
	hr = xAudio2->CreateMasteringVoice(&tmpXAudio2MasteringVoice,
		XAUDIO2_DEFAULT_CHANNELS, XAUDIO2_DEFAULT_SAMPLERATE,
		NULL, NULL, NULL);
	if (hr != S_OK)
	{
		// エラーメッセージ
		std::stringstream tmpMsg("");
		tmpMsg << "0x" << std::hex << hr;
		std::string errorMsg = errorNS::XAudio2CreateMastering + tmpMsg.str();
		MessageBox(NULL, errorMsg.c_str(), APP_NAME, (MB_OK | MB_ICONERROR));
		tmpMsg.clear();
		errorMsg.clear();
	}

	return tmpXAudio2MasteringVoice;
}

//===================================================================================================================================
// ボイスリソースの作成
//===================================================================================================================================
void XAudio2Manager::CreateVoiceResourceVoice(IXAudio2 *xAudio2, std::string voiceName, SoundResource soundResource)
{
	IXAudio2 *tmpXAudio = xAudio2;	// 外部からのXAudio2
	IXAudio2SourceVoice *tmpXAudio2SourceVoice = nullptr;

	// 外部からのXAudio2が存在しないなら
	if (tmpXAudio == nullptr)
	{
		// 内部のXAudio2
		tmpXAudio = XAudio2;
	}
	
	// バッファの設定
	XAUDIO2_BUFFER buffer = { 0 };
	buffer.pAudioData = (BYTE*)soundResource.data;
	buffer.Flags = XAUDIO2_END_OF_STREAM;
	buffer.AudioBytes = soundResource.size;
	buffer.LoopCount = 255;
	buffer.LoopBegin = 0;

	// ソースボイスの作成
	tmpXAudio->CreateSourceVoice(&tmpXAudio2SourceVoice, &soundResource.waveFormatEx);
	tmpXAudio2SourceVoice->SubmitSourceBuffer(&buffer);


	// ボイスリソースの作成
	voiceResource[voiceName].sourceVoice = tmpXAudio2SourceVoice;
	voiceResource[voiceName].isPlaying = false;
}

//===================================================================================================================================
// ソースボイスの再生・一時停止
//===================================================================================================================================
void XAudio2Manager::PlayPauseSourceVoice(IXAudio2 *xAudio2, std::string voiceName)
{
	// 該当ボイスが存在しない
	if (voiceResource.count(voiceName) == NULL)
	{
		IXAudio2 *tmpXAudio = xAudio2;	// 外部からのXAudio2

		// 外部からのXAudio2が存在しないなら
		if (tmpXAudio == nullptr)
		{
			// 内部のXAudio2
			tmpXAudio = XAudio2;
		}

		// ソースボイスの作成
		CreateVoiceResourceVoice(tmpXAudio, voiceName, soundBase->soundResource[voiceName]);
	}

	// 再生
	if (!voiceResource[voiceName].isPlaying)
	{
		voiceResource[voiceName].sourceVoice->Start();
		voiceResource[voiceName].isPlaying = true;
	}
	else
	{
		voiceResource[voiceName].sourceVoice->Stop();
		voiceResource[voiceName].isPlaying = false;
	}
}

//===================================================================================================================================
// マスターボイスボリューム(レベル)の取得
//===================================================================================================================================
float XAudio2Manager::GetMasteringVoiceVolumeLevel(void)
{
	float volume = NULL;
	
	// ボリューム取得
	XAudio2MasteringVoice->GetVolume(&volume);

	return volume;
}

//===================================================================================================================================
// マスターボイスボリューム(レベル)の調整
//===================================================================================================================================
HRESULT XAudio2Manager::SetMasteringVoiceVolumeLevel(float _volume)
{
	HRESULT hr = E_FAIL;

	// 重い対策
	if (_volume != oldMasteringVoiceVolume)
	{
		// ボリューム調整
		if ((_volume <= xAudioManagerNS::overVolume) && (_volume >= xAudioManagerNS::minVolume))
		{
			hr = XAudio2MasteringVoice->SetVolume(_volume);
			oldMasteringVoiceVolume = _volume;
		}
	}
	return hr;
}

//===================================================================================================================================
// 再生状態
//===================================================================================================================================
bool XAudio2Manager::GetIsPlaying(std::string voiceName)
{
	if (voiceResource.count(voiceName) == NULL)
	{
		return false;
	}

	return voiceResource[voiceName].isPlaying;
}

//===================================================================================================================================
// ボイス状態
//===================================================================================================================================
XAUDIO2_VOICE_STATE XAudio2Manager::GetVoiceState(std::string voiceName)
{
	XAUDIO2_VOICE_STATE voiceState = { NULL };

	if (voiceResource.count(voiceName) > NULL)
	{
		voiceResource[voiceName].sourceVoice->GetState(&voiceState);
	}

	return voiceState;
}