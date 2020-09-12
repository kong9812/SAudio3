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
	tmpMixerParameter.maxSample = (float)soundBase->soundResource[mixResourceData.soundName].size / (float)sizeof(short) / (float)soundBase->soundResource[mixResourceData.soundName].waveFormatEx.nChannels;
	tmpMixerParameter.maxMs = ((float)tmpMixerParameter.maxSample / (float)soundBase->soundResource[mixResourceData.soundName].waveFormatEx.nSamplesPerSec) * 1000.0f;
	tmpMixerParameter.XAudio2SourceVoice = xAudio2Manager->CreateSourceVoice(mixResourceData.soundName);
	tmpMixerParameter.soundName = mixResourceData.soundName;
	tmpMixerParameter.parameterName = mixResourceData.soundName + std::to_string(mixResourceData.cnt);
	tmpMixerParameter.sAudio3FadeParameter.fadeInStartMs = NULL;
	tmpMixerParameter.sAudio3FadeParameter.fadeInEndMs = 1;
	tmpMixerParameter.sAudio3FadeParameter.fadeOutStartMs = tmpMixerParameter.maxMs - 1;
	tmpMixerParameter.sAudio3FadeParameter.fadeOutEndMs = tmpMixerParameter.maxMs;
	tmpMixerParameter.sAudio3FadeParameter.silentBeforeFade = false;
	tmpMixerParameter.sAudio3FadeParameter.silentAfterFade = false;
	tmpMixerParameter.isPlaying = false;
	tmpMixerParameter.isFade = false;
	tmpMixerParameter.playingPos = NULL;

	// �t�F�C�h�G�t�F�N�g�̐ݒu
	HRESULT hr = xAudio2Manager->SetXapoFade(tmpMixerParameter.XAudio2SourceVoice);
	tmpMixerParameter.sAudio3FadeParameter.allSampling = (float)tmpMixerParameter.maxSample * (float)soundBase->soundResource[mixResourceData.soundName].waveFormatEx.nChannels;
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
					int mixerParametersSizeX = tmpTextSize.x + 300;
					ImGui::SetColumnWidth(idx, mixerParametersSizeX);

					// [�p�[�c]�Đ��v���C���[
					MixerPartPlayer(i, tmpTextSize.y);

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
						// [�p�[�c]�~�N�T�[
						MixerPartMixer(i);

						ImGui::NextColumn();
						idx++;
						ImGui::PopID();

						i++;
					}
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
		if (ImGui::Button("Pause"))
		{
			mixerParameter->XAudio2SourceVoice->Stop();
			mixerParameter->isPlaying = !mixerParameter->isPlaying;
		}
	}
	else
	{
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
	ImGui::Checkbox("Silent before fade", &mixerParameter->sAudio3FadeParameter.silentBeforeFade);
	ImGui::SameLine();
	ImGui::Checkbox("Silent after fade", &mixerParameter->sAudio3FadeParameter.silentAfterFade);

	ImGui::PushItemWidth(200);
	int processingSample = NULL;
	mixerParameter->XAudio2SourceVoice->GetEffectParameters(0, &processingSample, sizeof(float));
	ImGui::TextDisabled("Processing Sampling(1ch):%d", (int)((float)processingSample / (float)soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels));
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::Text("Processing Ms(1ch):%.f", SAMPLE_TO_MS(((float)processingSample / (float)soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels), 
			soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec,
			soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels));
		ImGui::EndTooltip();
	}
	ImGui::ProgressBar(mixerParameter->playingPos, ImVec2(200, imGuiManagerNS::soundBasePanelProgressBarSize.y), "");

	// Fade In Start Pos(ms)
	if (ImGui::SliderInt("[Fade in] Start Pos(ms)", &mixerParameter->sAudio3FadeParameter.fadeInStartMs, 0.0f, mixerParameter->maxMs))
	{
		if ((mixerParameter->sAudio3FadeParameter.fadeInStartMs >= mixerParameter->sAudio3FadeParameter.fadeInEndMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeInStartMs >= mixerParameter->sAudio3FadeParameter.fadeOutStartMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeInStartMs >= mixerParameter->sAudio3FadeParameter.fadeOutEndMs))
		{
			mixerParameter->sAudio3FadeParameter.fadeInStartMs = oldMixerParameter.sAudio3FadeParameter.fadeInStartMs;
		}
	}

	// Fade In End Pos(ms)
	if (ImGui::SliderInt("[Fade in] End Pos(ms)", &mixerParameter->sAudio3FadeParameter.fadeInEndMs, 0.0f, mixerParameter->maxMs))
	{
		if ((mixerParameter->sAudio3FadeParameter.fadeInEndMs <= mixerParameter->sAudio3FadeParameter.fadeInStartMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeInEndMs >= mixerParameter->sAudio3FadeParameter.fadeOutStartMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeInEndMs >= mixerParameter->sAudio3FadeParameter.fadeOutEndMs))
		{
			mixerParameter->sAudio3FadeParameter.fadeInEndMs = oldMixerParameter.sAudio3FadeParameter.fadeInEndMs;
		}
	}

	// Fade Out Start Pos(ms)
	if (ImGui::SliderInt("[Fade out] Start Pos(ms)", &mixerParameter->sAudio3FadeParameter.fadeOutStartMs, 0.0f, mixerParameter->maxMs))
	{
		if ((mixerParameter->sAudio3FadeParameter.fadeOutStartMs >= mixerParameter->sAudio3FadeParameter.fadeOutEndMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeOutStartMs <= mixerParameter->sAudio3FadeParameter.fadeInStartMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeOutStartMs <= mixerParameter->sAudio3FadeParameter.fadeInEndMs))
		{
			mixerParameter->sAudio3FadeParameter.fadeOutStartMs = oldMixerParameter.sAudio3FadeParameter.fadeOutStartMs;
		}
	}

	// Fade Out End Pos(ms)
	if (ImGui::SliderInt("[Fade out] End Pos(ms)", &mixerParameter->sAudio3FadeParameter.fadeOutEndMs, 0.0f, mixerParameter->maxMs))
	{
		if ((mixerParameter->sAudio3FadeParameter.fadeOutEndMs <= mixerParameter->sAudio3FadeParameter.fadeOutStartMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeOutEndMs <= mixerParameter->sAudio3FadeParameter.fadeInStartMs) ||
			(mixerParameter->sAudio3FadeParameter.fadeOutEndMs <= mixerParameter->sAudio3FadeParameter.fadeInEndMs))
		{
			mixerParameter->sAudio3FadeParameter.fadeOutEndMs = oldMixerParameter.sAudio3FadeParameter.fadeOutEndMs;
		}
	}

	// ��r�E�ꊇ����
	if (memcmp(&oldMixerParameter.sAudio3FadeParameter, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter)) != 0)
	{
		// XAPO�̃p�����[�^�[�ݒu
		mixerParameter->XAudio2SourceVoice->SetEffectParameters(0, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter));
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