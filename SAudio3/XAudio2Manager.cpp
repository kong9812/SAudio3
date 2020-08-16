//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
XAudio2Manager::XAudio2Manager()
{
	HRESULT hr = NULL;

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
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
XAudio2Manager::~XAudio2Manager()
{
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
	HRESULT hr = NULL;
	IXAudio2MasteringVoice *tmpXAudio2MasteringVoice = nullptr;

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