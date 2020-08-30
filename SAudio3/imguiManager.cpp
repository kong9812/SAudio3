//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "ImguiManager.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
ImGuiManager::ImGuiManager(HWND hWnd, ID3D11Device *device,
	ID3D11DeviceContext	*deviceContext, TextureBase *_textureBase,
	SoundBase *_soundBase, XAudio2Manager *_xAudio2Manager)
{
	// �o�[�W�����`�F�b�N
	IMGUI_CHECKVERSION();

	// [ImGui]�R���e�N�X�g�̍쐬
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//ImFontConfig config;
	//config.MergeMode = true;
	//io.Fonts->AddFontDefault();
	io.Fonts->AddFontFromFileTTF("ImGui\\font\\utf-8.ttf", 15.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());

#if USE_IMGUI_DOCKING
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// �h�b�L���O�̎g�p����
#endif

	// �N���V�b�N���[�h(�F)
	ImGui::StyleColorsClassic();

	// [ImGui]win32�̏�����
	if (!ImGui_ImplWin32_Init(hWnd))
	{
		// �G���[���b�Z�[�W
		MessageBox(NULL, errorNS::ImGuiWin32InitError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

		// �����I��
		PostQuitMessage(0);
		return;
	}

	// [ImGui]������
	if (!ImGui_ImplDX11_Init(device, deviceContext))
	{
		// �G���[���b�Z�[�W
		MessageBox(NULL, errorNS::ImGuiInitError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

		// �����I��
		PostQuitMessage(0);
		return;
	}

	// ImGui�t���O�̏�����
	showMainPanel = true;
	showPlayerPanel = true;
	showSoundBasePanel = true;
	showMixerPanel = true;

	isPlaying = false;
	isMasteringVoiceVolumeOver1 = false;

	// �e�N�X�`���x�[�X
	textureBase = _textureBase;
	soundBase = _soundBase;
	xAudio2Manager = _xAudio2Manager;

	// �p�t�H�[�}���X�r���[�A�̏�����
	PerformanceViewerInit();

	// �v���b�g�}�l�[�W���[
	imGuiPlotManager = new ImGuiPlotManager;

	// �~�N�T�[�}�l�[�W���[
	imGuiMixerManager = new ImGuiMixerManager(_textureBase);
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
ImGuiManager::~ImGuiManager()
{
	// [ImGui]�I������
	SAFE_DELETE(imGuiPlotManager)
	SAFE_DELETE(imGuiMixerManager)
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

//===================================================================================================================================
// [ImGui]�V�����t���[���̍쐬
//===================================================================================================================================
void ImGuiManager::CreateNewFrame()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
}

//===================================================================================================================================
// [ImGui]���T�C�Y
//===================================================================================================================================
void ImGuiManager::ReSize(LONG right, LONG bottom)
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(right, bottom));
}

//===================================================================================================================================
// [ImGui]�w���v�}�[�N
//===================================================================================================================================
void ImGuiManager::HelpMarker(const char* desc)
{
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

//===================================================================================================================================
// [ImGui]���C���p�l��(���T�C�Y)
//===================================================================================================================================
void ImGuiManager::ShowPanel(bool reSize, RECT mainPanelSize)
{
	// ���T�C�Y
	if (reSize)
	{
		ReSize(mainPanelSize.right, mainPanelSize.bottom);
	}

	// ���C���p�l��
	MainPanel();
}

//===================================================================================================================================
// [ImGui]���C���p�l��
//===================================================================================================================================
void ImGuiManager::ShowPanel()
{
	// ���C���p�l��
	MainPanel();
}

//===================================================================================================================================
// ���C���p�l��
//===================================================================================================================================
void ImGuiManager::MainPanel()
{
	// �f�[�^�J�E���^�[���Z�b�g
	//imGuiPlotManager->ResetDataCnt();

	// ���C���p�l��
	if (showMainPanel)
	{
		ImGui::SetNextWindowBgAlpha(0.7f);
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("SAudio3", &showMainPanel,
			ImGuiWindowFlags_::ImGuiWindowFlags_MenuBar |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoBringToFrontOnFocus);

		// ���j���[�o�[
		MenuBar();

		// �e�X�g�����̕\��
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		// �p�t�H�[�}���X�r���[�A
		PerformanceViewer();

		// �h�b�L���O
		ImGuiID dockspaceID = ImGui::GetID("MainPanelDockSpace");
		ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
		ImGui::End();

	}

	// �Đ��p�l��
	PlayerPanel();

	// �T�E���h�x�[�X�p�l��
	SoundBasePanel();

	// �~�N�T�[�p�l��
	imGuiMixerManager->MixerPanel(&showMixerPanel);
}

//===================================================================================================================================
// �p�t�H�[�}���X�r���[�A�̏�����
//===================================================================================================================================
void ImGuiManager::PerformanceViewerInit()
{
	// CPU�R�A���̎擾
	GetSystemInfo(&performanceRecord.systeminfo);
	performanceRecord.cpuNum = performanceRecord.systeminfo.dwNumberOfProcessors;

	// �����������̎擾(4G�o�C�g�ȏ�̂�)
	performanceRecord.memoryStatusEx = { sizeof(MEMORYSTATUSEX) };
	GlobalMemoryStatusEx(&performanceRecord.memoryStatusEx);

	// �n���h���̍쐬
	PdhOpenQuery(NULL, 0, &performanceRecord.pdhHquery);
	
	// �A�v���P�[�V������
	std::string instanceName = "\\Process(SAudio3)";

	// �J�E���^�[�p�X�̓o�^
	std::string cpu_counter_path = instanceName + "\\% Processor Time";
	if (ERROR_SUCCESS != PdhAddCounter(performanceRecord.pdhHquery, cpu_counter_path.c_str(),
		0, &performanceRecord.cpuCounter)) 
	{
		performanceRecord.cpuCounter = nullptr;
	}
	std::string mem_counter_path = instanceName + "\\Working Set - Private";
	if (ERROR_SUCCESS != PdhAddCounter(performanceRecord.pdhHquery, mem_counter_path.c_str(),
		0, &performanceRecord.memoryCounter)) 
	{
		performanceRecord.memoryCounter = nullptr;
	}
}

//===================================================================================================================================
// �p�t�H�[�}���X�r���[�A
//===================================================================================================================================
void ImGuiManager::PerformanceViewer()
{
	// 1�b
	if ((timeGetTime() - performanceRecord.timeCnt) >= 1000)
	{
		performanceRecord.timeCnt = timeGetTime();
		PdhCollectQueryData(performanceRecord.pdhHquery);

		// CPU�g�p��
		PDH_FMT_COUNTERVALUE fmtvalue = { NULL };
		PDH_STATUS status = PdhGetFormattedCounterValue(performanceRecord.cpuCounter, PDH_FMT_DOUBLE, NULL, &fmtvalue);
		performanceRecord.cpuUsage = fmtvalue.doubleValue;

		// �����������
		status = PdhGetFormattedCounterValue(performanceRecord.memoryCounter, PDH_FMT_LONG, NULL, &fmtvalue);
		performanceRecord.memoryUsage = fmtvalue.longValue;
	}

	// CPU�̐�
	ImGui::Text(u8"CPU Core:%d    �����������̓��ڗe��:%.2f Gb",
		performanceRecord.cpuNum,
		BYTES_TO_GB(performanceRecord.memoryStatusEx.ullTotalPhys));

	// CPU�g�p��
	ImGui::Text(u8"CPU�g�p��:%.2f", performanceRecord.cpuUsage);

	// �����������
	ImGui::Text(u8"������:%.2f gb", BYTES_TO_GB(performanceRecord.memoryUsage));
}

//===================================================================================================================================
// ���j���[�o�[
//===================================================================================================================================
void ImGuiManager::MenuBar()
{
	if (ImGui::BeginMenuBar())
	{
		// �t�@�C������
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("New")){}
			ImGui::EndMenu();
		}

		// �E�C���h
		if (ImGui::BeginMenu("Window"))
		{
			ImGui::MenuItem("Player", "", &showPlayerPanel);
			ImGui::MenuItem("Sound Base", "", &showSoundBasePanel);	
			ImGui::MenuItem("Mixer", "", &showMixerPanel);		
			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}
}

//===================================================================================================================================
// �Đ��p�l��
//===================================================================================================================================
void ImGuiManager::PlayerPanel()
{
	// �Đ��p�l��
	if (showPlayerPanel)
	{
		ImGui::Begin("Player Panel", &showPlayerPanel, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);

		if (!isPlaying)
		{	
			// �Đ��{�^��
			if (ImGui::ImageButton((void*)textureBase->GetShaderResource((char *)"playButton.png"), imGuiManagerNS::buttonSize))
			{
				// �Đ�
				isPlaying = true;
			}
		}
		else
		{
			// �ꎞ��~�{�^��
			if (ImGui::ImageButton((void*)textureBase->GetShaderResource((char *)"pauseButton.png"), imGuiManagerNS::buttonSize))
			{
				// �ꎞ��~
				isPlaying = false;
			}
		}

		// �}�X�^�[�{�C�X�{�����[��
		MasteringVoiceVolumePanel();

		ImGui::End();
	}
}

//===================================================================================================================================
// �}�X�^�[�{�C�X�{�����[��
//===================================================================================================================================
void ImGuiManager::MasteringVoiceVolumePanel()
{
	// �}�X�^�[�{�C�X�{�����[��
	ImGui::Checkbox("-Over Volume-", &isMasteringVoiceVolumeOver1);
	ImGui::SameLine();
	HelpMarker(u8"�{�����[����1���傫���ݒ肵�������Ƀ`�F�b�N�����Ă�������"
		u8"(�����ꂪ�������邩������܂���)");

	float tmpVolume = xAudio2Manager->GetMasteringVoiceVolumeLevel();
	if (tmpVolume == xAudioManagerNS::minVolume)
	{
		ImGui::Text("Main Volume(Mastering Voice Volume(Level))              Volume(db):MIN");
	}
	else
	{
		ImGui::Text("Main Volume(Mastering Voice Volume(Level))              Volume(db):%.3f",
			XAudio2AmplitudeRatioToDecibels(tmpVolume));
	}

	// �X���C�_�[�̒�������
	if (ImGui::GetWindowSize().x > imGuiManagerNS::masteringVoiceVolumeSliderWidth)
	{
		ImGui::PushItemWidth(imGuiManagerNS::masteringVoiceVolumeSliderWidth);
	}

	// �I�[�o�[�{�����[��
	if (isMasteringVoiceVolumeOver1)
	{
		// 10�{�܂�
		ImGui::SliderFloat("", &tmpVolume, xAudioManagerNS::minVolume, xAudioManagerNS::overVolume);
	}
	else
	{
		// maxVolume�Ɏ��܂�
		if (tmpVolume > xAudioManagerNS::maxVolume)
		{
			tmpVolume = xAudioManagerNS::maxVolume;
		}
		ImGui::SliderFloat("", &tmpVolume, xAudioManagerNS::minVolume, xAudioManagerNS::maxVolume);
	}
	xAudio2Manager->SetMasteringVoiceVolumeLevel(tmpVolume);
}

//===================================================================================================================================
// �T�E���h�x�[�X�p�l��
//===================================================================================================================================
void ImGuiManager::SoundBasePanel()
{
	// �T�E���h�x�[�X�p�l��
	if (showSoundBasePanel)
	{
		ImGui::Begin("Sound Base Panel", &showSoundBasePanel, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);

		// �T�E���h���̕\��&�{�^��
		auto begin = soundBase->soundResource.begin();
		auto end = soundBase->soundResource.end();
		for (auto i = begin; i != end; i++)
		{
			// �`�F�b�N�{�b�N�X
			bool oldIsMix = i->second.isMix;
			ImGui::PushID(i->first.c_str());
			ImGui::Checkbox("Mix", &i->second.isMix);
			ImGui::PopID();
			ImGui::SameLine();
			if (oldIsMix != i->second.isMix)
			{
				imGuiMixerManager->SetMixerResource(i->first, i->second.isMix);
			}

			// �Đ����̂�
			if (xAudio2Manager->GetIsPlaying(i->first.data()))
			{
				// �Đ��{�^��
				if (ImGui::ImageButtonEx(ImGui::GetID(i->first.c_str()), (void*)textureBase->GetShaderResource((char *)"pauseButton.png"), imGuiManagerNS::buttonSize, ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0), ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
				{
					// �Đ��E�ꎞ��~
					xAudio2Manager->PlayPauseSourceVoice(nullptr, i->first.data());
				}
			}
			else
			{
				// �ꎞ��~�{�^��
				if (ImGui::ImageButtonEx(ImGui::GetID(i->first.c_str()), (void*)textureBase->GetShaderResource((char *)"playButton.png"), imGuiManagerNS::buttonSize, ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0), ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
				{
					// �Đ��E�ꎞ��~
					xAudio2Manager->PlayPauseSourceVoice(nullptr, i->first.data());
				}
			}
			// �T�E���h��
			ImGui::SameLine();
			ImGui::Text("%s", i->first.data());

			// �Đ��ʒu
			XAUDIO2_VOICE_STATE voiceState = xAudio2Manager->GetVoiceState(i->first.data());
			ImGui::SameLine();
			ImGui::Text("-Played Samples-:%d", voiceState.SamplesPlayed % (i->second.size / sizeof(short) / i->second.waveFormatEx.nChannels));
			float tmp = (voiceState.SamplesPlayed % (i->second.size / sizeof(short) / i->second.waveFormatEx.nChannels))
				/ (float)(i->second.size / sizeof(short) / i->second.waveFormatEx.nChannels);
			ImGui::ProgressBar(tmp, imGuiManagerNS::soundBasePanelProgressBarSize, "");

			//�v���b�g
			imGuiPlotManager->PlotCompressWave(i->first.data(), &i->second);
		}
		ImGui::End();
	}
}