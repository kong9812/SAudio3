//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "ImguiManager.h"
#include "ImGuiMixerManager.h"
#include "SampleRateNormalizer.h"
#include "cudaCalc.cuh"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
ImGuiMixerManager::ImGuiMixerManager(XAudio2Manager *_xAudio2Manager,TextureBase *_textureBase, SoundBase *_soundBase)
{
	xAudio2Manager = _xAudio2Manager;
	textureBase = _textureBase;
	soundBase = _soundBase;
	updataCombineSound = false;
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
			// [チャンネルごと]データ部の削除
			for (int j = 0; j < i.combineSoundData.channel; j++)
			{
				SAFE_DELETE_ARRAY(i.combineSoundData.data[j])
			}
			SAFE_DELETE_ARRAY(i.combineSoundData.data)

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
	tmpMixerParameter.combineSoundData = { NULL };

	// フェイドエフェクトの設置
	HRESULT hr = xAudio2Manager->SetXapoFade(tmpMixerParameter.XAudio2SourceVoice);
	tmpMixerParameter.sAudio3FadeParameter.allSampling = (float)tmpMixerParameter.maxSample * (float)soundBase->soundResource[mixResourceData.soundName].waveFormatEx.nChannels;
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
					int mixerParametersSizeX = tmpTextSize.x + 300;
					ImGui::SetColumnWidth(idx, mixerParametersSizeX);

					// [パーツ]再生プレイヤー
					MixerPartPlayer(i, tmpTextSize.y);

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
						// [パーツ]ミクサー
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
			// サウンドの合成
			MixerCombine();
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
	// 書き出し
	//bool adpcm = false;	// ADPCM
	bool output = false;	// Normal

	ImGui::Checkbox("Silent before fade", &mixerParameter->sAudio3FadeParameter.silentBeforeFade);
	ImGui::SameLine();
	ImGui::Checkbox("Silent after fade", &mixerParameter->sAudio3FadeParameter.silentAfterFade);

	// 書き出し
	//ImGui::Checkbox("[Not Available]ADPCM Out", &adpcm);
	ImGui::Checkbox("Output", &output);

	// 書き出し
	if (output)
	{
		MixWithOutCUDA(*mixerParameter);
	}
	//if (adpcm)
	//{
	//	soundBase->OutputSound(soundBase->soundResource[mixerParameter->soundName].data,
	//		soundBase->soundResource[mixerParameter->soundName].size,
	//		soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nChannels,
	//		soundBase->soundResource[mixerParameter->soundName].waveFormatEx.nSamplesPerSec, true);
	//}


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

	// 比較・一括処理
	if (memcmp(&oldMixerParameter.sAudio3FadeParameter, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter)) != 0)
	{
		// XAPOのパラメーター設置
		mixerParameter->XAudio2SourceVoice->SetEffectParameters(0, &mixerParameter->sAudio3FadeParameter, sizeof(SAudio3FadeParameter));
		updataCombineSound = true;
	}
}

//===================================================================================================================================
// サウンドの合成
//===================================================================================================================================
void ImGuiMixerManager::MixerCombine(void)
{
	// 更新があれば
	if (updataCombineSound)
	{
		long finalSampingPerChannel = NULL;	// 最後のサンプリング数
		int maxChannel = NULL;				// 最大のチャンネル数

		// GPU計算
		CUDA_CALC *cudaCalc = new CUDA_CALC;

		// サウンド合成の準備(正規化・フェイド・無音を除く)
		for (auto i = mixerData.mixerParameter.begin(); i != mixerData.mixerParameter.end(); i++)
		{
			i->combineSoundData.channel = soundBase->soundResource[i->soundName].waveFormatEx.nChannels;

			// 正規化
			Normalize_Data normalizeData = cudaCalc->normalizer(soundBase->soundResource[i->soundName].data,
				soundBase->soundResource[i->soundName].size, soundBase->soundResource[i->soundName].waveFormatEx.nChannels,
				soundBase->soundResource[i->soundName].waveFormatEx.nSamplesPerSec, 48000, 1.5f);
			
			soundBase->OutputSound(normalizeData.newData, normalizeData.newSize,
				soundBase->soundResource[i->soundName].waveFormatEx.nChannels, 48000);

			// フェイド
			Fade_Data fadeData = cudaCalc->fade(normalizeData,
				soundBase->soundResource[i->soundName].waveFormatEx.nChannels,
				i->sAudio3FadeParameter);

			// float変換
			i->combineSoundData.data = new float*[i->combineSoundData.channel];
			Conversion_Data tmpConversionData = cudaCalc->conversion(fadeData.newData, fadeData.newSize,
				i->combineSoundData.channel);
			i->combineSoundData.sampingPerChannel = tmpConversionData.sampingPerChannel;

			for (int j = 0; j < soundBase->soundResource[i->soundName].waveFormatEx.nChannels; j++)
			{
				i->combineSoundData.data[j] = new float[tmpConversionData.sampingPerChannel];

				memcpy(&i->combineSoundData.data[j][0],
					&tmpConversionData.data[j][0],
					tmpConversionData.sampingPerChannel * sizeof(float));

				SAFE_DELETE_ARRAY(tmpConversionData.data[j])
			}
			SAFE_DELETE_ARRAY(tmpConversionData.data)

			// サイズ・チャンネル数
			finalSampingPerChannel += i->combineSoundData.sampingPerChannel;
			if (maxChannel < i->combineSoundData.channel)
			{
				maxChannel = i->combineSoundData.channel;
			}

			// 後片付け
			SAFE_DELETE(fadeData.newData)
			SAFE_DELETE(normalizeData.newData)
		}

		// チャンネル合成
		for (auto i = mixerData.mixerParameter.begin(); i != mixerData.mixerParameter.end(); i++)
		{
			short *finalData = new short[i->combineSoundData.sampingPerChannel*maxChannel];
			finalData = cudaCalc->combine(i->combineSoundData.data,
				i->combineSoundData.sampingPerChannel,
				i->combineSoundData.channel,
				maxChannel);

			long newSize = i->combineSoundData.sampingPerChannel*maxChannel * sizeof(short);

			//soundBase->OutputSound(finalData, newSize, maxChannel, 44100);
			SAFE_DELETE_ARRAY(finalData)

			//SAFE_DELETE_ARRAY(finalCombineSound);
			//finalCombineSound = new short[finalSampingPerChannel*maxChannel];
		}

		SAFE_DELETE(cudaCalc);

		updataCombineSound = false;
	}

	//// プロット
	//for (auto i = combineSoundData.begin(); i != combineSoundData.end(); i++)
	//{
	//	for (int j = 0; j < i->channel; j++)
	//	{
	//		ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
	//		ImGui::PlotLines("", i->data[j], i->sampingPerChannel, 0, "", FLT_MAX, FLT_MAX, plotextent);
	//	}
	//}
}

//===================================================================================================================================
// サウンドの合成(CUDA抜き)
//===================================================================================================================================
void ImGuiMixerManager::MixWithOutCUDA(Mixer_Parameter mixerParameter)
{
	SoundResource soundResource = soundBase->soundResource[mixerParameter.soundName];

	// フェイドイン
	int fadeInStartSampling = MS_TO_SAMPLING(mixerParameter.sAudio3FadeParameter.fadeInStartMs, soundResource.waveFormatEx.nSamplesPerSec*soundResource.waveFormatEx.nChannels);
	int fadeInEndSampling = MS_TO_SAMPLING(mixerParameter.sAudio3FadeParameter.fadeInEndMs, soundResource.waveFormatEx.nSamplesPerSec*soundResource.waveFormatEx.nChannels);
	float fadeAddVolume = 1.0f / ((float)fadeInEndSampling - (float)fadeInStartSampling);
	// フェイドアウト
	int fadeOutStartSampling = MS_TO_SAMPLING(mixerParameter.sAudio3FadeParameter.fadeOutStartMs, soundResource.waveFormatEx.nSamplesPerSec*soundResource.waveFormatEx.nChannels);
	int fadeOutEndSampling = MS_TO_SAMPLING(mixerParameter.sAudio3FadeParameter.fadeOutEndMs, soundResource.waveFormatEx.nSamplesPerSec*soundResource.waveFormatEx.nChannels);
	float fadeMinusVolume = 1.0f / ((float)fadeOutEndSampling - (float)fadeOutStartSampling);

	short *outputBuf = new short[mixerParameter.maxSample * soundResource.waveFormatEx.nChannels];

	for (long i = 0; i < mixerParameter.maxSample * soundResource.waveFormatEx.nChannels; i++)
	{
		// 初期化から
		outputBuf[i] = soundResource.data[i];

		// フェイドイン処理
		if ((i >= fadeInStartSampling) && (i <= fadeInEndSampling))
		{
			// フェイド中の位置
			int fadeIdx = i - fadeInStartSampling;

			// ボリューム計算
			float volume = (fadeAddVolume)*fadeIdx;
			outputBuf[i] = soundResource.data[i] * volume;
		}
		else if ((i >= fadeOutStartSampling) && (i <= fadeOutEndSampling))
		{
			// フェイド中の位置
			int fadeIdx = i - fadeOutStartSampling;

			// ボリューム計算
			float volume = 1.0f - ((fadeMinusVolume)*fadeIdx);
			outputBuf[i] = soundResource.data[i] * volume;
		}

		// 無音処理
		if ((mixerParameter.sAudio3FadeParameter.silentBeforeFade) && (i < fadeInStartSampling))
		{
			outputBuf[i] = 0.0f;
		}
		else if ((mixerParameter.sAudio3FadeParameter.silentAfterFade) && (i > fadeOutEndSampling))
		{
			outputBuf[i] = 0.0f;
		}
	}

	soundBase->OutputSound(outputBuf,
		soundResource.size,
		soundResource.waveFormatEx.nChannels,
		soundResource.waveFormatEx.nSamplesPerSec, false);

	SAFE_DELETE(outputBuf);
}