//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "ImguiManager.h"
#include "ImGuiMixerManager.h"
#include "SampleRateNormalizer.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
ImGuiMixerManager::ImGuiMixerManager(XAudio2Manager *_xAudio2Manager,TextureBase *_textureBase, SoundBase *_soundBase)
{
	xAudio2Manager = _xAudio2Manager;
	textureBase = _textureBase;
	soundBase = _soundBase;
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
ImGuiMixerManager::~ImGuiMixerManager()
{
	// ���X�g�̍폜
	if (mixerData.mixerResource.size() > NULL)
	{
		mixerData.mixerResource.clear();
	}
	// ���X�g�̍폜
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
// �~�N�T�[���\�[�X�̐ݒu
// bool addUse: true=�ǉ� false=�폜
//===================================================================================================================================
void ImGuiMixerManager::SetMixerResource(std::string soundName,bool addUse)
{
	Mixer_Resource tmpData;
	tmpData.soundName = soundName;
	tmpData.cnt = NULL;

	// �ǉ�
	if (addUse)
	{
		// ���X�g�̖����ɒǉ�
		mixerData.mixerResource.push_back(tmpData);
	}
	// �폜
	else
	{
		// ���X�g����
		int idx = 0;
		// �v�f�̍폜(�����_)
		mixerData.mixerResource.remove_if([=](Mixer_Resource x) 
		{ return x.soundName == tmpData.soundName; });
	}
}

//===================================================================================================================================
// �~�N�T�[�p�����[�^�[�̍쐬
//===================================================================================================================================
Mixer_Parameter ImGuiMixerManager::CreateMixerParameter(Mixer_Resource mixResourceData)
{
	// �~�N�T�[�p�����[�^�[�̏�����
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

	// �t�F�[�h�G�t�F�N�g�̐ݒu
	HRESULT hr = xAudio2Manager->SetXapoFade(tmpMixerParameter.XAudio2SourceVoice);
	tmpMixerParameter.sAudio3FadeParameter.allSampling = soundBase->soundResource[mixResourceData.soundName].size /
		sizeof(short);
	hr = tmpMixerParameter.XAudio2SourceVoice->SetEffectParameters(0, &tmpMixerParameter.sAudio3FadeParameter, sizeof(SAudio3FadeParameter));
	return tmpMixerParameter;
}

//===================================================================================================================================
// �~�N�T�[�p�l��
//===================================================================================================================================
void ImGuiMixerManager::MixerPanel(bool *showMixerPanael)
{
	if (showMixerPanael)
	{
		ImGui::Begin("Mixer Panel", showMixerPanael, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);
		// �~�N�T�[���\�[�X
		if (ImGui::BeginChild("Mixer Resource"))
		{
			// �~�N�T�[���\�[�X�ꗗ
			for (std::list<Mixer_Resource>::iterator i = mixerData.mixerResource.begin(); i != mixerData.mixerResource.end(); i++)
			{
				ImGui::SameLine();

				// �{�^�����������牺�ɒǉ�����
				if (ImGui::Button(i->soundName.c_str()))
				{
					// �~�N�T�[�p�����[�^�[�̍쐬
					mixerData.mixerParameter.push_back(CreateMixerParameter(*i));

					// �~�N�T�[�p�����[�^�[�̒ǉ�
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

				// �~�N�T�[�p�����[�^�[�ꗗ
				int idx = 1;
				for (auto i = mixerData.mixerParameter.begin(); i != mixerData.mixerParameter.end();)
				{
					ImGui::PushID(i->parameterName.c_str());
					ImVec2 tmpTextSize = ImGui::CalcTextSize(i->parameterName.c_str());
					int mixerParametersSizeX = tmpTextSize.x + 250;
					ImGui::SetColumnWidth(idx, mixerParametersSizeX);

					// [�p�[�c]�Đ��v���C���[
					MixerPartPlayer(i, tmpTextSize.y);

					// [�p�[�c]�~�N�T�[
					MixerPartMixer(i);

					ImGui::SameLine();

					// [�p�[�c]�폜�{�^��
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
// [�p�[�c]�Đ��v���C���[
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

	// �Đ��ʒu
	XAUDIO2_VOICE_STATE voiceState = { NULL };
	mixerParameter->XAudio2SourceVoice->GetState(&voiceState);
	mixerParameter->playingPos = ((voiceState.SamplesPlayed % (soundBase->soundResource[mixerParameter->soundName].size / sizeof(short) / soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels))
		/ (float)(soundBase->soundResource[mixerParameter->soundName].size / sizeof(short) / soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels));
	int playingMs = voiceState.SamplesPlayed % (soundBase->soundResource[mixerParameter->soundName].size / sizeof(short) / soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels) /
		(float)soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec * 1000.0f;
	// �T�E���h��
	ImGui::TextDisabled("%s Playing:%0.f%%", mixerParameter->parameterName.c_str(), mixerParameter->playingPos * 100);
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::Text("Playing:%dms", playingMs);
		ImGui::EndTooltip();
	}
}

//===================================================================================================================================
// [�p�[�c]�폜�{�^��
//===================================================================================================================================
bool ImGuiMixerManager::MixerPartDelete(std::list<Mixer_Parameter>::iterator mixerParameter, bool deleteButton)
{
	// ��Ƀ{�C�X�̍폜���s��
	if (deleteButton)
	{
		SAFE_DESTROY_VOICE(mixerParameter->XAudio2SourceVoice)
	}
	return deleteButton;
}

//===================================================================================================================================
// [�p�[�c]�~�N�T�[
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
		ImGui::Text(u8"Sampling:1ch������̃T���v�����O��");
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
		ImGui::Text(u8"Sampling:1ch������̃T���v�����O��");
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
			ImGui::Text(u8"Sampling:1ch������̃T���v�����O��");
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
			ImGui::Text(u8"Sampling:1ch������̃T���v�����O��");
			ImGui::EndTooltip();
		}
	}

	// ��r�E�ꊇ����
	if (memcmp(&oldMixerParameter.sAudio3FadeParameter, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter)) != 0)
	{
		// XAPO�̃p�����[�^�[�ݒu
		// mixerParameter->XAudio2SourceVoice->SetEffectParameters(0, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter));
	}
}

////===================================================================================================================================
//// ���M�f�B�X�N���v�^�̍쐬�E�ݒu
//// �~�N�T�[�p�����[�^�[�̃T�E���h���͌��ɐ��������Ă�
//// ��:xxx.wav0 , xxx.wav1
////===================================================================================================================================
//void ImGuiMixerManager::SetSendDescriptor(std::string mixerParameterName,
//	std::list<Mixer_Resource>::iterator mixerResource, IXAudio2SubmixVoice *XAudio2SubmixVoice)
//{
//	mixerResource->sendDescriptor[mixerParameterName].Flags = NULL;
//	mixerResource->sendDescriptor[mixerParameterName].pOutputVoice = XAudio2SubmixVoice;
//}