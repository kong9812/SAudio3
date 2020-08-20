//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
XAudio2Manager::XAudio2Manager(SoundBase *_soundBase)
{
	HRESULT hr = E_FAIL;

	// COM�̏�����
	hr = (CoInitializeEx(nullptr, COINIT_MULTITHREADED));
	if (hr != S_OK)
	{
		// �G���[���b�Z�[�W
		std::stringstream tmpMsg("");
		tmpMsg << "0x" << std::hex << hr;
		std::string errorMsg = errorNS::XAudio2ComInitError + tmpMsg.str();
		MessageBox(NULL, errorMsg.c_str(), APP_NAME, (MB_OK | MB_ICONERROR));
		tmpMsg.clear();
		errorMsg.clear();

		// �����I��
		PostQuitMessage(0);
	}

	// XAudio2�̏�����
#if ( _WIN32_WINNT < _WIN32_WINNT_WIN8) && defined(_DEBUG)
	// Win7�̃f�o�b�O�@�\
	hr = XAudio2Create(&XAudio2, XAUDIO2_DEBUG_ENGINE, XAUDIO2_DEFAULT_PROCESSOR);
#else
	hr = XAudio2Create(&XAudio2, NULL, XAUDIO2_DEFAULT_PROCESSOR);
#endif
	if (hr != S_OK)
	{
		// �G���[���b�Z�[�W
		std::stringstream tmpMsg("");
		tmpMsg << "0x" << std::hex << hr;
		std::string errorMsg = errorNS::XAudio2InitError + tmpMsg.str();
		MessageBox(NULL, errorMsg.c_str(), APP_NAME, (MB_OK | MB_ICONERROR));
		tmpMsg.clear();
		errorMsg.clear();

		// �����I��
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
	// Win7�ȍ~�̃f�o�b�O�@�\
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

	// �}�X�^�[�{�C�X�̍쐬
	XAudio2MasteringVoice = CreateMasterVoice(XAudio2);
	if (XAudio2MasteringVoice == nullptr)
	{
		// �����I��
		PostQuitMessage(0);
	}

	soundBase = _soundBase;
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
XAudio2Manager::~XAudio2Manager()
{
	// �A�z�z��̍폜
	if (voiceResource.size() > NULL)
	{
		// �T�E���h���\�[�X�̏I������
		auto begin = voiceResource.begin();
		auto end = voiceResource.end();
		for (auto i = begin; i != end; i++)
		{
			// �\�[�X�{�C�X�̏I������
			//i->second.sourceVoice->Stop();
			SAFE_DESTROY_VOICE(i->second.sourceVoice)
		}
		voiceResource.clear();
	}

	// �}�X�^�[�{�C�X�̏I������
	SAFE_DESTROY_VOICE(XAudio2MasteringVoice)

	// XAudio2�̏I������
	SAFE_RELEASE(XAudio2)
	
	// COM�̏I������
	CoUninitialize();
}

//===================================================================================================================================
// �}�X�^�[�{�C�X�̍쐬
//===================================================================================================================================
IXAudio2MasteringVoice *XAudio2Manager::CreateMasterVoice(IXAudio2 *xAudio2)
{
	HRESULT hr = E_FAIL;
	IXAudio2MasteringVoice *tmpXAudio2MasteringVoice = nullptr;

	// �}�X�^�[�{�C�X�̍쐬
	hr = xAudio2->CreateMasteringVoice(&tmpXAudio2MasteringVoice,
		XAUDIO2_DEFAULT_CHANNELS, XAUDIO2_DEFAULT_SAMPLERATE,
		NULL, NULL, NULL);
	if (hr != S_OK)
	{
		// �G���[���b�Z�[�W
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
// �{�C�X���\�[�X�̍쐬
//===================================================================================================================================
void XAudio2Manager::CreateVoiceResourceVoice(IXAudio2 *xAudio2, std::string voiceName, SoundResource soundResource)
{
	IXAudio2 *tmpXAudio = xAudio2;	// �O�������XAudio2
	IXAudio2SourceVoice *tmpXAudio2SourceVoice = nullptr;

	// �O�������XAudio2�����݂��Ȃ��Ȃ�
	if (tmpXAudio == nullptr)
	{
		// ������XAudio2
		tmpXAudio = XAudio2;
	}
	
	// �o�b�t�@�̐ݒ�
	XAUDIO2_BUFFER buffer = { 0 };
	buffer.pAudioData = (BYTE*)soundResource.data;
	buffer.Flags = XAUDIO2_END_OF_STREAM;
	buffer.AudioBytes = soundResource.size;
	buffer.LoopCount = 255;
	buffer.LoopBegin = 0;

	// �\�[�X�{�C�X�̍쐬
	tmpXAudio->CreateSourceVoice(&tmpXAudio2SourceVoice, &soundResource.waveFormatEx);
	tmpXAudio2SourceVoice->SubmitSourceBuffer(&buffer);


	// �{�C�X���\�[�X�̍쐬
	voiceResource[voiceName].sourceVoice = tmpXAudio2SourceVoice;
	voiceResource[voiceName].isPlaying = false;
}

//===================================================================================================================================
// �\�[�X�{�C�X�̍Đ��E�ꎞ��~
//===================================================================================================================================
void XAudio2Manager::PlayPauseSourceVoice(IXAudio2 *xAudio2, std::string voiceName)
{
	// �Y���{�C�X�����݂��Ȃ�
	if (voiceResource.count(voiceName) == NULL)
	{
		IXAudio2 *tmpXAudio = xAudio2;	// �O�������XAudio2

		// �O�������XAudio2�����݂��Ȃ��Ȃ�
		if (tmpXAudio == nullptr)
		{
			// ������XAudio2
			tmpXAudio = XAudio2;
		}

		// �\�[�X�{�C�X�̍쐬
		CreateVoiceResourceVoice(tmpXAudio, voiceName, soundBase->soundResource[voiceName]);
	}

	// �Đ�
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
// �}�X�^�[�{�C�X�{�����[��(���x��)�̎擾
//===================================================================================================================================
float XAudio2Manager::GetMasteringVoiceVolumeLevel(void)
{
	float volume = NULL;
	
	// �{�����[���擾
	XAudio2MasteringVoice->GetVolume(&volume);

	return volume;
}

//===================================================================================================================================
// �}�X�^�[�{�C�X�{�����[��(���x��)�̒���
//===================================================================================================================================
HRESULT XAudio2Manager::SetMasteringVoiceVolumeLevel(float _volume)
{
	HRESULT hr = E_FAIL;

	// �d���΍�
	if (_volume != oldMasteringVoiceVolume)
	{
		// �{�����[������
		if ((_volume <= xAudioManagerNS::overVolume) && (_volume >= xAudioManagerNS::minVolume))
		{
			hr = XAudio2MasteringVoice->SetVolume(_volume);
			oldMasteringVoiceVolume = _volume;
		}
	}
	return hr;
}

//===================================================================================================================================
// �Đ����
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
// �{�C�X���
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