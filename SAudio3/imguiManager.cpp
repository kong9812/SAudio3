//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "ImguiManager.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
ImGuiManager::ImGuiManager(HWND hWnd, ID3D11Device *device,
	ID3D11DeviceContext	*deviceContext, TextureBase *_textureBase,
	SoundBase *_soundBase, XAudio2Manager *_xAudio2Manager)
{
	// バージョンチェック
	IMGUI_CHECKVERSION();

	// [ImGui]コンテクストの作成
	ImGui::CreateContext();
	ImPlot::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//ImFontConfig config;
	//config.MergeMode = true;
	//io.Fonts->AddFontDefault();
	io.Fonts->AddFontFromFileTTF("ImGui\\font\\utf-8.ttf", 15.0f, nullptr, io.Fonts->GetGlyphRangesJapanese());

#if USE_IMGUI_DOCKING
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// ドッキングの使用許可
#endif

	// GUIの初期化
	ImGui::StyleColorsClassic();
	ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_GrabMinSize, 18.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_FrameRounding, 12.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_::ImGuiStyleVar_GrabRounding, 12.0f);
	
	// [ImGui]win32の初期化
	if (!ImGui_ImplWin32_Init(hWnd))
	{
		// エラーメッセージ
		MessageBox(NULL, errorNS::ImGuiWin32InitError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

		// 強制終了
		PostQuitMessage(0);
		return;
	}

	// [ImGui]初期化
	if (!ImGui_ImplDX11_Init(device, deviceContext))
	{
		// エラーメッセージ
		MessageBox(NULL, errorNS::ImGuiInitError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

		// 強制終了
		PostQuitMessage(0);
		return;
	}

	// ImGuiフラグの初期化
	showMainPanel = true;
	showPlayerPanel = true;
	showSoundBasePanel = true;
	showMixerPanel = true;
	showImDemo = false;

	isPlaying = false;
	isMasteringVoiceVolumeOver1 = false;

	// テクスチャベース
	textureBase = _textureBase;
	soundBase = _soundBase;
	xAudio2Manager = _xAudio2Manager;

	// パフォーマンスビューアの初期化
	PerformanceViewerInit();

	// プロットマネージャー
	imGuiPlotManager = new ImGuiPlotManager;

	// ミクサーマネージャー
	imGuiMixerManager = new ImGuiMixerManager(xAudio2Manager,_textureBase, _soundBase);
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
ImGuiManager::~ImGuiManager()
{
	// [ImGui]終了処理
	SAFE_DELETE(imGuiPlotManager)
	SAFE_DELETE(imGuiMixerManager)
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
	ImPlot::DestroyContext();
}

//===================================================================================================================================
// [ImGui]新しいフレームの作成
//===================================================================================================================================
void ImGuiManager::CreateNewFrame()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
}

//===================================================================================================================================
// [ImGui]リサイズ
//===================================================================================================================================
void ImGuiManager::ReSize(LONG right, LONG bottom)
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(right, bottom));
}

//===================================================================================================================================
// [ImGui]ヘルプマーク
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
// [ImGui]メインパネル(リサイズ)
//===================================================================================================================================
void ImGuiManager::ShowPanel(bool reSize, RECT mainPanelSize)
{
	// リサイズ
	if (reSize)
	{
		ReSize(mainPanelSize.right, mainPanelSize.bottom);
	}

	// メインパネル
	MainPanel();
}

//===================================================================================================================================
// [ImGui]メインパネル
//===================================================================================================================================
void ImGuiManager::ShowPanel()
{
	// メインパネル
	MainPanel();
}

//===================================================================================================================================
// メインパネル
//===================================================================================================================================
void ImGuiManager::MainPanel()
{
#ifdef _DEBUG
	// デモ画面
	if (showImDemo)
	{
		ImPlot::ShowDemoWindow(&showImDemo);
		ImGui::ShowDemoWindow(&showImDemo);
	}
#endif

	// データカウンターリセット
	//imGuiPlotManager->ResetDataCnt();

	// メインパネル
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

		// メニューバー
		MenuBar();

		// テスト文字の表示
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		// パフォーマンスビューア
		PerformanceViewer();

		// 処理サンプリング(44100・48000)


		// ドッキング
		ImGuiID dockspaceID = ImGui::GetID("MainPanelDockSpace");
		ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
		ImGui::End();

	}

	// 再生パネル
	PlayerPanel();

	// サウンドベースパネル
	SoundBasePanel();

	// ミクサーパネル
	imGuiMixerManager->MixerPanel(&showMixerPanel);
}

//===================================================================================================================================
// パフォーマンスビューアの初期化
//===================================================================================================================================
void ImGuiManager::PerformanceViewerInit()
{
	// CPUコア数の取得
	GetSystemInfo(&performanceRecord.systeminfo);
	performanceRecord.cpuNum = performanceRecord.systeminfo.dwNumberOfProcessors;

	// 物理メモリの取得(4Gバイト以上のみ)
	performanceRecord.memoryStatusEx = { sizeof(MEMORYSTATUSEX) };
	GlobalMemoryStatusEx(&performanceRecord.memoryStatusEx);

	// ハンドルの作成
	PdhOpenQuery(NULL, 0, &performanceRecord.pdhHquery);
	
	// アプリケーション名
	std::string instanceName = "\\Process(SAudio3)";

	// カウンターパスの登録
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
// パフォーマンスビューア
//===================================================================================================================================
void ImGuiManager::PerformanceViewer()
{
	// 1秒
	if ((timeGetTime() - performanceRecord.timeCnt) >= 1000)
	{
		performanceRecord.timeCnt = timeGetTime();
		PdhCollectQueryData(performanceRecord.pdhHquery);

		// CPU使用率
		PDH_FMT_COUNTERVALUE fmtvalue = { NULL };
		PDH_STATUS status = PdhGetFormattedCounterValue(performanceRecord.cpuCounter, PDH_FMT_DOUBLE, NULL, &fmtvalue);
		performanceRecord.cpuUsage = fmtvalue.doubleValue;

		// メモリ消費量
		status = PdhGetFormattedCounterValue(performanceRecord.memoryCounter, PDH_FMT_LONG, NULL, &fmtvalue);
		performanceRecord.memoryUsage = fmtvalue.longValue;
	}

	// CPUの数
	ImGui::Text(u8"CPU Core:%d    物理メモリの搭載容量:%.2f Gb",
		performanceRecord.cpuNum,
		BYTES_TO_GB(performanceRecord.memoryStatusEx.ullTotalPhys));

	// CPU使用率
	ImGui::Text(u8"CPU使用率:%.2f", performanceRecord.cpuUsage);

	// メモリ消費量
	ImGui::Text(u8"メモリ:%.2f gb", BYTES_TO_GB(performanceRecord.memoryUsage));
}

//===================================================================================================================================
// メニューバー
//===================================================================================================================================
void ImGuiManager::MenuBar()
{
	if (ImGui::BeginMenuBar())
	{
		// ファイル操作
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("New")){}
			ImGui::EndMenu();
		}

		// ウインド
		if (ImGui::BeginMenu("Window"))
		{
			ImGui::MenuItem("ImPlot Demo", "", &showImDemo);
			ImGui::MenuItem("Player", "", &showPlayerPanel);
			ImGui::MenuItem("Sound Base", "", &showSoundBasePanel);	
			ImGui::MenuItem("Mixer", "", &showMixerPanel);		
			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}
}

//===================================================================================================================================
// 再生パネル
//===================================================================================================================================
void ImGuiManager::PlayerPanel()
{
	// 再生パネル
	if (showPlayerPanel)
	{
		ImGui::Begin("Player Panel", &showPlayerPanel, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);

		if (!isPlaying)
		{	
			// 再生ボタン
			if (ImGui::ImageButton((void*)textureBase->GetShaderResource((char *)"playButton.png"), imGuiManagerNS::buttonSize))
			{
				// 再生
				isPlaying = true;
			}
		}
		else
		{
			// 一時停止ボタン
			if (ImGui::ImageButton((void*)textureBase->GetShaderResource((char *)"pauseButton.png"), imGuiManagerNS::buttonSize))
			{
				// 一時停止
				isPlaying = false;
			}
		}

		// マスターボイスボリューム
		MasteringVoiceVolumePanel();

		ImGui::End();
	}
}

//===================================================================================================================================
// マスターボイスボリューム
//===================================================================================================================================
void ImGuiManager::MasteringVoiceVolumePanel()
{
	// マスターボイスボリューム
	ImGui::Checkbox("-Over Volume-", &isMasteringVoiceVolumeOver1);
	ImGui::SameLine();
	HelpMarker(u8"ボリュームを1より大きく設定したい時にチェックを入れてください"
		u8"(音割れが発生するかもしれません)");

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

	// スライダーの長さ調整
	if (ImGui::GetWindowSize().x > imGuiManagerNS::masteringVoiceVolumeSliderWidth)
	{
		ImGui::PushItemWidth(imGuiManagerNS::masteringVoiceVolumeSliderWidth);
	}

	// オーバーボリューム
	if (isMasteringVoiceVolumeOver1)
	{
		// 10倍まで
		ImGui::SliderFloat("", &tmpVolume, xAudioManagerNS::minVolume, xAudioManagerNS::overVolume);
	}
	else
	{
		// maxVolumeに収まる
		if (tmpVolume > xAudioManagerNS::maxVolume)
		{
			tmpVolume = xAudioManagerNS::maxVolume;
		}
		ImGui::SliderFloat("", &tmpVolume, xAudioManagerNS::minVolume, xAudioManagerNS::maxVolume);
	}
	xAudio2Manager->SetMasteringVoiceVolumeLevel(tmpVolume);
}

//===================================================================================================================================
// サウンドベースパネル
//===================================================================================================================================
void ImGuiManager::SoundBasePanel()
{
	// サウンドベースパネル
	if (showSoundBasePanel)
	{
		ImGui::Begin("Sound Base Panel", &showSoundBasePanel, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);

		// サウンド名の表示&ボタン
		auto begin = soundBase->soundResource.begin();
		auto end = soundBase->soundResource.end();
		for (auto i = begin; i != end; i++)
		{
			// チェックボックス
			bool oldIsMix = i->second.isMix;
			ImGui::PushID(i->first.c_str());
			ImGui::Checkbox("Mix", &i->second.isMix);

#if SAUDIO3_TEST_VER
			if (ImGui::Button("[TEST]Filter"))
			{
				ImGui::OpenPopup("TEST window");

				if (!xAudio2Manager->GetIsPlaying(i->first.data()))
				{
					// 再生・一時停止
					xAudio2Manager->PlayPauseSourceVoice(nullptr, i->first.data());
				}

				IXAudio2SourceVoice *sourceVoice = xAudio2Manager->GetSourceVoice(i->first.data());
				XAUDIO2_EFFECT_DESCRIPTOR	effectDescriptor = { NULL };	// エフェクトディスクリプタ
				XAUDIO2_EFFECT_CHAIN		chain = { NULL };				// エフェクトチェン
				XAUDIO2_VOICE_DETAILS		voiceDetails = { NULL };		// ボイス詳細
				sourceVoice->GetVoiceDetails(&voiceDetails);				// ボイス詳細の取得

				// XAPOs
				IUnknown *XApo = (IXAPO *)new SAudio3FilterXapo();

				// エフェクトディスクリプタの初期化
				effectDescriptor.pEffect = XApo;
				effectDescriptor.InitialState = true;
				effectDescriptor.OutputChannels = voiceDetails.InputChannels;

				// エフェクトチェンの初期化
				chain.EffectCount = 1;
				chain.pEffectDescriptors = &effectDescriptor;

				// ソースボイスとの接続
				sourceVoice->SetEffectChain(&chain);

				// パラメーター
				SAudio3FilterParameter filterParameter;
				filterParameter.type = XAPO_FILTER_TYPE::XFT_LowpassFilter;
				filterParameter.Q = 0.5f;
				filterParameter.cutoffFreq = (voiceDetails.InputSampleRate / 2) / 2;
				filterParameter.bandwidth = 1.f;

				sourceVoice->SetEffectParameters(0, &filterParameter, sizeof(SAudio3FilterParameter));

				sourceVoice->Start();

				// すぐぽい！(たぶん大丈夫…確認待ち)
				SAFE_RELEASE(XApo);

				lastFrameSample = 0;
				lastFramePlayedSamples = 0;
				writePos = 0;
				inL = new kiss_fft_cpx[i->second.waveFormatEx.nSamplesPerSec];
				out = new kiss_fft_cpx[i->second.waveFormatEx.nSamplesPerSec];
				FFT_R = new float[i->second.waveFormatEx.nSamplesPerSec];
			}

			ImVec2 center = ImGui::GetMainViewport()->GetCenter();
			ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

			if (ImGui::BeginPopupModal("TEST window", NULL, ImGuiWindowFlags_::ImGuiWindowFlags_AlwaysAutoResize))
			{
				ImGui::Text("[TEST]Filter");
				const char *name = i->first.data();

				IXAudio2SourceVoice *sourceVoice = xAudio2Manager->GetSourceVoice(name);
				XAUDIO2_VOICE_STATE voiceState = xAudio2Manager->GetVoiceState(name);
				XAUDIO2_VOICE_DETAILS voiceDetails = xAudio2Manager->GetVoiceDetails(name);

				// パラメーター
				SAudio3FilterParameter filterParameter;
				sourceVoice->GetEffectParameters(0, &filterParameter, sizeof(SAudio3FilterParameter));
				SAudio3FilterParameter newFilterParameter = filterParameter;

				ImGui::Text("Name: %s", name);
				ImGui::Text("0:Lowpass  1:Highpass  2:Bandpass  3:Notch");
				ImGui::SliderInt("Type", (int *)&newFilterParameter.type, 0, 3);
				ImGui::SliderInt("Cutoff Freq.", &newFilterParameter.cutoffFreq, 1, (voiceDetails.InputSampleRate / 2));
				ImGui::SliderFloat("Q", &newFilterParameter.Q, .1f, 5.f);
				ImGui::SliderFloat("Bandwidth", &newFilterParameter.bandwidth, .1f, 5.f);

				if ((newFilterParameter.type != filterParameter.type) ||
					(newFilterParameter.cutoffFreq != filterParameter.cutoffFreq) ||
					(newFilterParameter.Q != filterParameter.Q) ||
					(newFilterParameter.bandwidth != filterParameter.bandwidth))
				{
					sourceVoice->SetEffectParameters(0, &newFilterParameter, sizeof(SAudio3FilterParameter));
				}

				ImGui::Spacing();
				ImGui::Text(u8"[TEST]FFT");

				long currentSample = voiceState.SamplesPlayed;
				ImGui::Text("Samples pre frame: %d", lastFramePlayedSamples);

				if ((currentSample - lastFrameSample) != 0)
				{
					lastFramePlayedSamples = currentSample - lastFrameSample;					

					Conversion_Data conversionData = imGuiPlotManager->GetConversionData(name);

					int size = i->second.size / sizeof(short) / i->second.waveFormatEx.nChannels;
					int readPos = currentSample;
					int startPos = (readPos - i->second.waveFormatEx.nSamplesPerSec) % size;
					if (startPos < 0)
					{
						startPos = 0;
					}

					for (int j = 0; j < i->second.waveFormatEx.nSamplesPerSec; j++)
					{
						if ((startPos + j) <= readPos)
						{
							inL[j].r = conversionData.data[0][(startPos + j) % size];
						}
						else
						{
							inL[j].r = 0;
						}
						inL[j].i = 0;
					}

					FFT::FFTProcess(i->second.waveFormatEx.nSamplesPerSec, inL, out);

					for (int j = 0; j < i->second.waveFormatEx.nSamplesPerSec; j++)
					{
						FFT_R[j] = out[j].r > 1 ? out[j].r : 0;
					}
				}

				ImVec2 plotextent(1500, 100);
				ImGui::PlotHistogram("", FFT_R, i->second.waveFormatEx.nSamplesPerSec / 2, 0, "", 0, FLT_MAX, plotextent);

				lastFrameSample = currentSample;

				if (ImGui::Button("Close"))
				{
					SAFE_DELETE_ARRAY(inL);
					SAFE_DELETE_ARRAY(out);
					SAFE_DELETE_ARRAY(FFT_R);
					sourceVoice->SetEffectChain(NULL);
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndPopup();
			}
#endif

			ImGui::PopID();
			ImGui::SameLine();
			if (oldIsMix != i->second.isMix)
			{
				imGuiMixerManager->SetMixerResource(i->first, i->second.isMix);
			}

			// 再生中のみ
			if (xAudio2Manager->GetIsPlaying(i->first.data()))
			{
				// 再生ボタン
				if (ImGui::ImageButtonEx(ImGui::GetID(i->first.c_str()), (void*)textureBase->GetShaderResource((char *)"pauseButton.png"), imGuiManagerNS::buttonSize, ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0), ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
				{
					// 再生・一時停止
					xAudio2Manager->PlayPauseSourceVoice(nullptr, i->first.data());
				}
			}
			else
			{
				// 一時停止ボタン
				if (ImGui::ImageButtonEx(ImGui::GetID(i->first.c_str()), (void*)textureBase->GetShaderResource((char *)"playButton.png"), imGuiManagerNS::buttonSize, ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0), ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
				{
					// 再生・一時停止
					xAudio2Manager->PlayPauseSourceVoice(nullptr, i->first.data());
				}
			}
			// サウンド名
			ImGui::SameLine();
			ImGui::Text("%s", i->first.data());

			// 再生位置
			XAUDIO2_VOICE_STATE voiceState = xAudio2Manager->GetVoiceState(i->first.data());
			ImGui::SameLine();
			ImGui::Text("-Played Samples-:%d", voiceState.SamplesPlayed % (i->second.size / sizeof(short) / i->second.waveFormatEx.nChannels));
			float tmp = (voiceState.SamplesPlayed % (i->second.size / sizeof(short) / i->second.waveFormatEx.nChannels))
				/ (float)(i->second.size / sizeof(short) / i->second.waveFormatEx.nChannels);
			ImGui::ProgressBar(tmp, imGuiManagerNS::soundBasePanelProgressBarSize, "");
			
			XAUDIO2_VOICE_DETAILS voiceDetails = xAudio2Manager->GetVoiceDetails(i->first.data());
			if (voiceDetails.InputChannels != NULL)
			{
				ImGui::Text("-Input Channels-:%d   -Input SampleRate-:%d", voiceDetails.InputChannels, voiceDetails.InputSampleRate);
			}

			//プロット
			imGuiPlotManager->PlotCompressWave(i->first.data(), &i->second);
		}
		ImGui::End();
	}
}