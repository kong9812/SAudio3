//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
XAudio2Manager::XAudio2Manager(SoundBase *_soundBase)
{
	HRESULT hr = S_OK;

	//// COMの初期化
	//hr = (CoInitializeEx(nullptr, COINIT_MULTITHREADED));
	if (hr != S_OK)
	{
		// エラーメッセージ
		std::stringstream tmpMsg("");
		tmpMsg << "0x" << std::hex << hr;
		std::string errorMsg = errorNS::XAudio2ComInitError + tmpMsg.str();
		MessageBox(NULL, errorMsg.c_str(), MAIN_APP_NAME, (MB_OK | MB_ICONERROR));
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
		MessageBox(NULL, errorMsg.c_str(), MAIN_APP_NAME, (MB_OK | MB_ICONERROR));
		tmpMsg.clear();
		errorMsg.clear();

		// 強制終了
		PostQuitMessage(0);
	}

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

	// エフェクトマネージャーの初期化
	xAudio2EffectManager = new XAudio2EffectManager;

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
			XAUDIO2_VOICE_SENDS sendList = { NULL };
			HRESULT hr = i->second.sourceVoice->SetOutputVoices(&sendList);
			SAFE_DESTROY_VOICE(i->second.sourceVoice)
		}
		voiceResource.clear();
	}
	// エフェクトマネージャーの終了処理
	SAFE_DELETE(xAudio2EffectManager)

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
		MessageBox(NULL, errorMsg.c_str(), MAIN_APP_NAME, (MB_OK | MB_ICONERROR));
		tmpMsg.clear();
		errorMsg.clear();
	}

	return tmpXAudio2MasteringVoice;
}

//===================================================================================================================================
// ボイスリソースの作成
//===================================================================================================================================
HRESULT XAudio2Manager::CreateVoiceResourceVoice(IXAudio2 *xAudio2, std::string voiceName, SoundResource soundResource)
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
	HRESULT hr = tmpXAudio->CreateSourceVoice(&tmpXAudio2SourceVoice, &soundResource.waveFormatEx);
	hr = tmpXAudio2SourceVoice->SubmitSourceBuffer(&buffer);


	// ボイスリソースの作成
	voiceResource[voiceName].sourceVoice = tmpXAudio2SourceVoice;
	voiceResource[voiceName].isPlaying = false;

	return hr;
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

//===================================================================================================================================
// ボイス詳細
//===================================================================================================================================
XAUDIO2_VOICE_DETAILS XAudio2Manager::GetVoiceDetails(std::string voiceName)
{
	XAUDIO2_VOICE_DETAILS voiceDetails = { NULL };

	if (voiceResource.count(voiceName) > NULL)
	{
		voiceResource[voiceName].sourceVoice->GetVoiceDetails(&voiceDetails);
	}
	return voiceDetails;
}

//===================================================================================================================================
// 処理サンプリングの設定
//===================================================================================================================================
void XAudio2Manager::SetProcessSampling(int _processSampling)
{
	processSampling = _processSampling;
}

//===================================================================================================================================
// 処理サンプリングの取得
//===================================================================================================================================
int XAudio2Manager::GetProcessSampling(void)
{
	return processSampling;
}

//===================================================================================================================================
// サブミックスボイスの作成
//===================================================================================================================================
IXAudio2SubmixVoice *XAudio2Manager::CreateSubmixVoice(std::string voiceName)
{
	IXAudio2SubmixVoice *tmpSubmixVoice = nullptr;
	XAudio2->CreateSubmixVoice(&tmpSubmixVoice, soundBase->soundResource[voiceName].waveFormatEx.nChannels,
		soundBase->soundResource[voiceName].waveFormatEx.nSamplesPerSec);
	return tmpSubmixVoice;
}

//===================================================================================================================================
// サブミックスボイスの作成
//===================================================================================================================================
IXAudio2SourceVoice *XAudio2Manager::CreateSourceVoice(std::string voiceName)
{
	IXAudio2SourceVoice *tmpXAudio2SourceVoice = nullptr;

	// バッファの設定
	XAUDIO2_BUFFER buffer = { 0 };
	buffer.pAudioData = (BYTE*)soundBase->soundResource[voiceName].data;
	buffer.Flags = XAUDIO2_END_OF_STREAM;
	buffer.AudioBytes = soundBase->soundResource[voiceName].size;
	buffer.LoopCount = 255;
	buffer.LoopBegin = 0;

	// ソースボイスの作成
	HRESULT hr = XAudio2->CreateSourceVoice(&tmpXAudio2SourceVoice, &soundBase->soundResource[voiceName].waveFormatEx);
	hr = tmpXAudio2SourceVoice->SubmitSourceBuffer(&buffer);

	return tmpXAudio2SourceVoice;
}

//===================================================================================================================================
// アウトプットボイスの設定
//===================================================================================================================================
HRESULT XAudio2Manager::SetOutputVoice(std::string voiceName, 
	std::map <std::string, XAUDIO2_SEND_DESCRIPTOR> _sendDescriptorList, int sendCount)
{
	// 連想配列から全要素の取り出し(本当はやりたくないが…)
	XAUDIO2_SEND_DESCRIPTOR *sendDescriptorList = new XAUDIO2_SEND_DESCRIPTOR[sendCount];
	auto begin = _sendDescriptorList.begin();
	auto end = _sendDescriptorList.end();
	int idx = 0;
	for (auto i = begin; i != end; i++, idx++)
	{
		sendDescriptorList[idx] = i->second;
	}

	XAUDIO2_VOICE_SENDS voiceSends = { NULL };
	voiceSends.SendCount = sendCount;
	voiceSends.pSends = sendDescriptorList;

	if (voiceResource[voiceName].sourceVoice == nullptr)
	{
		CreateVoiceResourceVoice(XAudio2, voiceName, soundBase->soundResource[voiceName]);
	}
	HRESULT hr = voiceResource[voiceName].sourceVoice->SetOutputVoices(&voiceSends);
	SAFE_DELETE(sendDescriptorList)
	return hr;
}