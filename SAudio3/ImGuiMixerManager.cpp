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
	tmpMixerParameter.XAudio2SourceVoice = xAudio2Manager->CreateSourceVoice(mixResourceData.soundName);
	tmpMixerParameter.soundName = mixResourceData.soundName + std::to_string(mixResourceData.cnt);
	tmpMixerParameter.fadeInMs = NULL;
	tmpMixerParameter.fadeInPos = NULL;
	tmpMixerParameter.fadeOutMs = NULL;
	tmpMixerParameter.fadeOutPos = NULL;
	tmpMixerParameter.isPlaying = false;
	tmpMixerParameter.isFade = false;

	tmpMixerParameter.playingPos = rand() % 10 / 10.0f;

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
				ImGui::BeginColumns("Mixer Parameters", mixerData.mixerParameter.size(),
					ImGuiColumnsFlags_::ImGuiColumnsFlags_GrowParentContentsSize);
				ImGui::Separator();

				// ミクサーパラメーター一覧
				int idx = 0;
				for (auto i = mixerData.mixerParameter.begin(); i != mixerData.mixerParameter.end();)
				{
					ImGui::PushID(i->soundName.c_str());
					ImVec2 tmpTextSize = ImGui::CalcTextSize(i->soundName.c_str());
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

	// サウンド名
	ImGui::Text(mixerParameter->soundName.c_str());
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

//===================================================================================================================================
// 送信ディスクリプタの作成・設置
// ミクサーパラメーターのサウンド名は後ろに数字がついてる
// 例:xxx.wav0 , xxx.wav1
//===================================================================================================================================
void ImGuiMixerManager::SetSendDescriptor(std::string mixerParameterName,
	std::list<Mixer_Resource>::iterator mixerResource, IXAudio2SubmixVoice *XAudio2SubmixVoice)
{
	mixerResource->sendDescriptor[mixerParameterName].Flags = NULL;
	mixerResource->sendDescriptor[mixerParameterName].pOutputVoice = XAudio2SubmixVoice;
}