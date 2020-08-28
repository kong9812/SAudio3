//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "ImguiManager.h"
#include "ImGuiMixerManager.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
ImGuiMixerManager::ImGuiMixerManager(TextureBase *_textureBase)
{
	textureBase = _textureBase;
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

	// 連想配列の削除
	if (mixerData.mixerParameter.size() > NULL)
	{
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
				for (std::list<Mixer_Parameter>::iterator i = mixerData.mixerParameter.begin(); i != mixerData.mixerParameter.end(); i++)
				{
					ImGui::PushID(i->soundName.c_str());
					ImVec2 tmpTextSize = ImGui::CalcTextSize(i->soundName.c_str());
					int mixerParametersSizeX = tmpTextSize.x + 250;

					ImGui::SetColumnWidth(idx, mixerParametersSizeX);
					if (i->isPlaying == true)
					{
						if (ImGui::ImageButtonEx(ImGui::GetID(i->soundName.c_str()), (void*)textureBase->GetShaderResource((char *)"pauseButton.png"), ImVec2(tmpTextSize.y, tmpTextSize.y), ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0), ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
						{
							i->isPlaying = false;
						}
					}
					else
					{
						if (ImGui::ImageButtonEx(ImGui::GetID(i->soundName.c_str()), (void*)textureBase->GetShaderResource((char *)"playButton.png"), ImVec2(tmpTextSize.y, tmpTextSize.y), ImVec2(-1, 0), ImVec2(0, 1), ImVec2(0, 0), ImVec4(1.0f, 1.0f, 1.0f, 0.0f), ImVec4(1.0f, 1.0f, 1.0f, 1.0f)))
						{
							i->isPlaying = true;
						}
					}
					ImGui::SameLine();
					ImGui::Text(i->soundName.c_str());
					ImGui::SameLine();
					ImGui::Button("Del");
					ImGui::PushItemWidth(200);
					ImGui::SliderInt("In Pos(S)", &i->fadeInPos, 0.0f, 200.0f);
					ImGui::ProgressBar(i->playingPos,
						ImVec2(200,imGuiManagerNS::soundBasePanelProgressBarSize.y), "");
					ImGui::SliderInt("Out Pos(S)", &i->fadeOutPos, 0.0f, 200.0f);
					ImGui::Checkbox("Cross Fade", &i->isFade);
					if (i->isFade)
					{
						ImGui::SliderInt("Fade In Time(S)", &i->fadeInMs, 0.0f, 200.0f);
						ImGui::SliderInt("Fade Out Time(S)", &i->fadeOutMs, 0.0f, 200.0f);
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
// ミクサーパラメーターの作成
//===================================================================================================================================
Mixer_Parameter ImGuiMixerManager::CreateMixerParameter(Mixer_Resource mixResourceData)
{
	// ミクサーパラメーターの初期化
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