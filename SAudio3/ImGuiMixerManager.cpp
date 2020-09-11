//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "ImguiManager.h"
#include "ImGuiMixerManager.h"
#include "SampleRateNormalizer.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
ImGuiMixerManager::ImGuiMixerManager(XAudio2Manager *_xAudio2Manager,TextureBase *_textureBase, SoundBase *_soundBase)
{
	xAudio2Manager = _xAudio2Manager;
	textureBase = _textureBase;
	soundBase = _soundBase;
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
ImGuiMixerManager::~ImGuiMixerManager()
{
	// リストの削除
	if (mixerData.mixerResource.size() > NULL)
	{
		mixerData.mixerResource.clear();
	}
	// リストの削除
	if (mixerData.mixerParameter.size() > NULL)
	{
		for (auto i : mixerData.mixerParameter)
		{
			SAFE_DESTROY_VOICE(i.XAudio2SourceVoice)
		}
		mixerData.mixerParameter.clear();
	}
}

//===================================================================================================================================
// ミクサーリソースの設置
// bool addUse: true=追加 false=削除
//===================================================================================================================================
void ImGuiMixerManager::SetMixerResource(std::string soundName,bool addUse)
{
	Mixer_Resource tmpData;
	tmpData.soundName = soundName;
	tmpData.cnt = NULL;

	// 追加
	if (addUse)
	{
		// リストの末尾に追加
		mixerData.mixerResource.push_back(tmpData);
	}
	// 削除
	else
	{
		// リスト検索
		int idx = 0;
		// 要素の削除(ラムダ)
		mixerData.mixerResource.remove_if([=](Mixer_Resource x) 
		{ return x.soundName == tmpData.soundName; });
	}
}

//===================================================================================================================================
// ミクサーパラメーターの作成
//===================================================================================================================================
Mixer_Parameter ImGuiMixerManager::CreateMixerParameter(Mixer_Resource mixResourceData)
{
	// ミクサーパラメーターの初期化
	Mixer_Parameter tmpMixerParameter;
	tmpMixerParameter.maxSample = soundBase->soundResource[mixResourceData.soundName].size / sizeof(short) / soundBase->soundResource[mixResourceData.soundName].waveFormatEx.nChannels;
	tmpMixerParameter.maxMs = ((soundBase->soundResource[mixResourceData.soundName].size / sizeof(short) / soundBase->soundResource[mixResourceData.soundName].waveFormatEx.nChannels) /
		soundBase->soundResource[mixResourceData.soundName].waveFormatEx.nSamplesPerSec) * 1000;
	tmpMixerParameter.XAudio2SourceVoice = xAudio2Manager->CreateSourceVoice(mixResourceData.soundName);
	tmpMixerParameter.soundName = mixResourceData.soundName;
	tmpMixerParameter.parameterName = mixResourceData.soundName + std::to_string(mixResourceData.cnt);
	tmpMixerParameter.sAudio3FadeParameter.fadeInMs = NULL;
	tmpMixerParameter.sAudio3FadeParameter.fadeInPosMs = NULL;
	tmpMixerParameter.sAudio3FadeParameter.fadeOutMs = NULL;
	tmpMixerParameter.sAudio3FadeParameter.fadeOutPosMs = tmpMixerParameter.maxMs;
	tmpMixerParameter.isPlaying = false;
	tmpMixerParameter.isFade = false;
	tmpMixerParameter.playingPos = NULL;

	// フェードエフェクトの設置
	HRESULT hr = xAudio2Manager->SetXapoFade(tmpMixerParameter.XAudio2SourceVoice);
	tmpMixerParameter.sAudio3FadeParameter.allSampling = soundBase->soundResource[mixResourceData.soundName].size /
		sizeof(short);
	hr = tmpMixerParameter.XAudio2SourceVoice->SetEffectParameters(0, &tmpMixerParameter.sAudio3FadeParameter, sizeof(SAudio3FadeParameter));
	return tmpMixerParameter;
}

//===================================================================================================================================
// ミクサーパネル
//===================================================================================================================================
void ImGuiMixerManager::MixerPanel(bool *showMixerPanael)
{
	if (showMixerPanael)
	{
		ImGui::Begin("Mixer Panel", showMixerPanael, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);
		// ミクサーリソース
		if (ImGui::BeginChild("Mixer Resource"))
		{
			// ミクサーリソース一覧
			for (std::list<Mixer_Resource>::iterator i = mixerData.mixerResource.begin(); i != mixerData.mixerResource.end(); i++)
			{
				ImGui::SameLine();

				// ボタンを押したら下に追加する
				if (ImGui::Button(i->soundName.c_str()))
				{
					// ミクサーパラメーターの作成
					mixerData.mixerParameter.push_back(CreateMixerParameter(*i));

					// ミクサーパラメーターの追加
					i->cnt++;
				}
			}
		}

		if (mixerData.mixerParameter.size() > NULL)
		{
			if (ImGui::BeginChild("Mixer Parameter", ImVec2(0, 0), false, ImGuiWindowFlags_::ImGuiWindowFlags_HorizontalScrollbar))
			{
				ImGui::BeginColumns("Mixer Parameters", mixerData.mixerParameter.size() + 1,
					ImGuiColumnsFlags_::ImGuiColumnsFlags_GrowParentContentsSize |
					ImGuiColumnsFlags_::ImGuiColumnsFlags_NoResize);
				ImGui::Separator();
				ImGui::SetColumnWidth(0, 20);
				ImGui::NextColumn();

				// ミクサーパラメーター一覧
				int idx = 1;
				for (auto i = mixerData.mixerParameter.begin(); i != mixerData.mixerParameter.end();)
				{
					ImGui::PushID(i->parameterName.c_str());
					ImVec2 tmpTextSize = ImGui::CalcTextSize(i->parameterName.c_str());
					int mixerParametersSizeX = tmpTextSize.x + 250;
					ImGui::SetColumnWidth(idx, mixerParametersSizeX);

					// [パーツ]再生プレイヤー
					MixerPartPlayer(i, tmpTextSize.y);

					// [パーツ]ミクサー
					MixerPartMixer(i);

					ImGui::SameLine();

					// [パーツ]削除ボタン
					if (MixerPartDelete(i, ImGui::Button("Del")))
					{
						mixerData.mixerParameter.erase(i);
						ImGui::NextColumn();
						ImGui::PopID();
						break;
					}
					else
					{
						i++;
					}
					ImGui::NextColumn();
					idx++;
					ImGui::PopID();
				}
				ImGui::NextColumn();
				ImGui::EndColumns();
				ImGui::Separator();
			}
			ImGui::EndChild();
		}
		ImGui::EndChild();
		ImGui::End();
	}
}

//===================================================================================================================================
// [パーツ]再生プレイヤー
//===================================================================================================================================
void ImGuiMixerManager::MixerPartPlayer(std::list<Mixer_Parameter>::iterator mixerParameter, float buttomSize)
{
	if (mixerParameter->isPlaying)
	{
		//if (ImGui::ImageButtonEx(ImGui::GetID(mixerParameter->soundName.c_str()),
		//	(void*)textureBase->GetShaderResource((char *)"pauseButton.png"),
		//	ImVec2(buttomSize, buttomSize), ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0),
		//	ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
		if (ImGui::Button("Pause"))
		{
			mixerParameter->XAudio2SourceVoice->Stop();
			mixerParameter->isPlaying = !mixerParameter->isPlaying;
		}
	}
	else
	{
		//if (ImGui::ImageButtonEx(ImGui::GetID(mixerParameter->soundName.c_str()),
		//	(void*)textureBase->GetShaderResource((char *)"playButton.png"),
		//	ImVec2(buttomSize, buttomSize), ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0),
		//	ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
		if (ImGui::Button("Play"))
		{
			mixerParameter->XAudio2SourceVoice->Start();
			mixerParameter->isPlaying = !mixerParameter->isPlaying;
		}
	}
	ImGui::SameLine();

	// 再生位置
	XAUDIO2_VOICE_STATE voiceState = { NULL };
	mixerParameter->XAudio2SourceVoice->GetState(&voiceState);
	mixerParameter->playingPos = ((voiceState.SamplesPlayed % (soundBase->soundResource[mixerParameter->soundName].size / sizeof(short) / soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels))
		/ (float)(soundBase->soundResource[mixerParameter->soundName].size / sizeof(short) / soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels));
	int playingMs = voiceState.SamplesPlayed % (soundBase->soundResource[mixerParameter->soundName].size / sizeof(short) / soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels) /
		(float)soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec * 1000.0f;
	// サウンド名
	ImGui::TextDisabled("%s Playing:%0.f%%", mixerParameter->parameterName.c_str(), mixerParameter->playingPos * 100);
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::Text("Playing:%dms", playingMs);
		ImGui::EndTooltip();
	}
}

//===================================================================================================================================
// [パーツ]削除ボタン
//===================================================================================================================================
bool ImGuiMixerManager::MixerPartDelete(std::list<Mixer_Parameter>::iterator mixerParameter, bool deleteButton)
{
	// 先にボイスの削除を行う
	if (deleteButton)
	{
		SAFE_DESTROY_VOICE(mixerParameter->XAudio2SourceVoice)
	}
	return deleteButton;
}

//===================================================================================================================================
// [パーツ]ミクサー
//===================================================================================================================================
void ImGuiMixerManager::MixerPartMixer(std::list<Mixer_Parameter>::iterator mixerParameter)
{
	Mixer_Parameter oldMixerParameter = *mixerParameter;

	ImGui::PushItemWidth(200);
	int processingSample = NULL;
	mixerParameter->XAudio2SourceVoice->GetEffectParameters(0, &processingSample, sizeof(float));
	ImGui::TextDisabled("Processing Sampling:%d", processingSample);
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::Text("Processing Ms:%.f", SAMPLE_TO_MS(processingSample,
			soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec,
			soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels));
		ImGui::EndTooltip();
	}

	// In Pos(ms)
	ImGui::SliderInt("In Pos(ms)", &mixerParameter->sAudio3FadeParameter.fadeInPosMs, 0.0f, mixerParameter->maxMs);
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::Text("In Pos(sample):%.f",
			MS_TO_SAMPLING(mixerParameter->sAudio3FadeParameter.fadeInPosMs,
				soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec));
		ImGui::Text(u8"Sampling:1ch当たりのサンプリング数");
		ImGui::EndTooltip();
	}

	ImGui::ProgressBar(mixerParameter->playingPos, ImVec2(200, imGuiManagerNS::soundBasePanelProgressBarSize.y), "");

	// Out Pos(ms)
	ImGui::SliderInt("Out Pos(ms)", &mixerParameter->sAudio3FadeParameter.fadeOutPosMs, 0.0f, mixerParameter->maxMs);
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::Text("Out Pos(sample):%.f",
			MS_TO_SAMPLING(mixerParameter->sAudio3FadeParameter.fadeOutPosMs,
				soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec));
		ImGui::Text(u8"Sampling:1ch当たりのサンプリング数");
		ImGui::EndTooltip();
	}

	ImGui::Checkbox("Cross Fade", &mixerParameter->isFade);
	if (mixerParameter->isFade)
	{
		// Fade In Time(ms)
		ImGui::SliderInt("Fade In Time(ms)", &mixerParameter->sAudio3FadeParameter.fadeInMs,
			0.0f, mixerParameter->sAudio3FadeParameter.fadeOutPosMs - mixerParameter->sAudio3FadeParameter.fadeInPosMs);
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Fade In Time(sample):%.f",
				MS_TO_SAMPLING(mixerParameter->sAudio3FadeParameter.fadeInMs,
					soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec));
			ImGui::Text(u8"Sampling:1ch当たりのサンプリング数");
			ImGui::EndTooltip();
		}

		// Fade Out Time(ms)
		ImGui::SliderInt("Fade Out Time(ms)", &mixerParameter->sAudio3FadeParameter.fadeOutMs,
			0.0f, mixerParameter->maxMs - mixerParameter->sAudio3FadeParameter.fadeOutPosMs);
		if (ImGui::IsItemHovered())
		{
			ImGui::BeginTooltip();
			ImGui::Text("Fade Out Time(sample):%.f",
				MS_TO_SAMPLING(mixerParameter->sAudio3FadeParameter.fadeOutMs,
					soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec));
			ImGui::Text(u8"Sampling:1ch当たりのサンプリング数");
			ImGui::EndTooltip();
		}
	}

	// 比較・一括処理
	if (memcmp(&oldMixerParameter.sAudio3FadeParameter, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter)) != 0)
	{
		// XAPOのパラメーター設置
		// mixerParameter->XAudio2SourceVoice->SetEffectParameters(0, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter));
	}
}

////===================================================================================================================================
//// 送信ディスクリプタの作成・設置
//// ミクサーパラメーターのサウンド名は後ろに数字がついてる
//// 例:xxx.wav0 , xxx.wav1
////===================================================================================================================================
//void ImGuiMixerManager::SetSendDescriptor(std::string mixerParameterName,
//	std::list<Mixer_Resource>::iterator mixerResource, IXAudio2SubmixVoice *XAudio2SubmixVoice)
//{
//	mixerResource->sendDescriptor[mixerParameterName].Flags = NULL;
//	mixerResource->sendDescriptor[mixerParameterName].pOutputVoice = XAudio2SubmixVoice;
//}