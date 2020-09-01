//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "ImguiManager.h"
#include "ImGuiMixerManager.h"
#include "SampleRateNormalizer.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
ImGuiMixerManager::ImGuiMixerManager(TextureBase *_textureBase, SoundBase *_soundBase)
{
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

	// �A�z�z��̍폜
	if (mixerData.mixerParameter.size() > NULL)
	{
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
	Mixer_Parameter tmpData;
	tmpData.soundName = mixResourceData.soundName + std::to_string(mixResourceData.cnt);
	tmpData.fadeInMs = NULL;
	tmpData.fadeInPos = NULL;
	tmpData.fadeOutMs = NULL;
	tmpData.fadeOutPos = NULL;
	tmpData.isPlaying = false;
	tmpData.isFade = false;

	tmpData.playingPos = rand() % 10 / 10.0f;

	return tmpData;
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
				ImGui::BeginColumns("Mixer Parameters", mixerData.mixerParameter.size(),
					ImGuiColumnsFlags_::ImGuiColumnsFlags_GrowParentContentsSize);
				ImGui::Separator();

				// �~�N�T�[�p�����[�^�[�ꗗ
				int idx = 0;
				for (std::list<Mixer_Parameter>::iterator i = mixerData.mixerParameter.begin(); i != mixerData.mixerParameter.end(); i++)
				{
					ImGui::PushID(i->soundName.c_str());
					ImVec2 tmpTextSize = ImGui::CalcTextSize(i->soundName.c_str());
					int mixerParametersSizeX = tmpTextSize.x + 250;
					ImGui::SetColumnWidth(idx, mixerParametersSizeX);

					// [�p�[�c]�Đ��v���C���[
					MixerPartPlayer(i, tmpTextSize.y);

					ImGui::SameLine();

					// [�p�[�c]�폜�{�^��
					MixerPartDelete(ImGui::Button("Del"));

					// [�p�[�c]�~�N�T�[
					MixerPartMixer(i);

					ImGui::NextColumn();
					idx++;
					ImGui::PopID();
				}

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
		if (ImGui::ImageButtonEx(ImGui::GetID(mixerParameter->soundName.c_str()),
			(void*)textureBase->GetShaderResource((char *)"pauseButton.png"),
			ImVec2(buttomSize, buttomSize), ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0),
			ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
		{
			mixerParameter->isPlaying = !mixerParameter->isPlaying;
		}
	}
	else
	{
		if (ImGui::ImageButtonEx(ImGui::GetID(mixerParameter->soundName.c_str()),
			(void*)textureBase->GetShaderResource((char *)"playButton.png"),
			ImVec2(buttomSize, buttomSize), ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0),
			ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
		{
			mixerParameter->isPlaying = !mixerParameter->isPlaying;
		}
	}
	ImGui::SameLine();

	// �T�E���h��
	ImGui::Text(mixerParameter->soundName.c_str());
}

//===================================================================================================================================
// [�p�[�c]�폜�{�^��
//===================================================================================================================================
void ImGuiMixerManager::MixerPartDelete(bool deleteButton)
{

}

//===================================================================================================================================
// [�p�[�c]�~�N�T�[
//===================================================================================================================================
void ImGuiMixerManager::MixerPartMixer(std::list<Mixer_Parameter>::iterator mixerParameter)
{
	ImGui::PushItemWidth(200);
	ImGui::SliderFloat("In Pos(S)", &mixerParameter->fadeInPos, 0.0f, 200.0f);
	ImGui::ProgressBar(mixerParameter->playingPos,
		ImVec2(200, imGuiManagerNS::soundBasePanelProgressBarSize.y), "");
	ImGui::SliderFloat("Out Pos(S)", &mixerParameter->fadeOutPos, 0.0f, 200.0f);
	ImGui::Checkbox("Cross Fade", &mixerParameter->isFade);
	if (mixerParameter->isFade)
	{
		ImGui::SliderFloat("Fade In Time(ms)", &mixerParameter->fadeInMs, 0.0f, 200.0f);
		ImGui::SliderFloat("Fade Out Time(ms)", &mixerParameter->fadeOutMs, 0.0f, 200.0f);
	}
}