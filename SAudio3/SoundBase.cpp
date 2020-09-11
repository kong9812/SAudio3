//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "SoundBase.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
SoundBase::SoundBase()
{
	WIN32_FIND_DATA FindFileData;	// ファイルデータ
	int fileNum = 0;				// ファイル数

	for (int i = 0; i < FILE_FORMAT::FILE_MAX; i++)
	{
		// ファイル検索ハンドル
		HANDLE hFile = FindFirstFileEx(soundBaseNS::fileFormat[i], FindExInfoBasic, &FindFileData,
			FindExSearchNameMatch, NULL, NULL);

		// ファイル検索
		if (hFile != INVALID_HANDLE_VALUE)
		{
			// ファイルパス
			std::string path = "Sound\\";
			path += FindFileData.cFileName;

			// シェーダーリソースの作成
			if (!LoadSound(path.c_str(), &soundResource[FindFileData.cFileName],i))
			{
				// エラーメッセージ
				MessageBox(NULL, errorNS::SoundResourceError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

				// 強制終了
				PostQuitMessage(0);
			}

			// クリア
			path.empty();

			// ファイルカウンター
			fileNum++;

			while (FindNextFile(hFile, &FindFileData))
			{
				// ファイルパス
				path = "Sound\\";
				path += FindFileData.cFileName;

				// シェーダーリソースの作成
				if (!LoadSound(path.c_str(), &soundResource[FindFileData.cFileName],i))
				{
					// エラーメッセージ
					MessageBox(NULL, errorNS::SoundResourceError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

					// 強制終了
					PostQuitMessage(0);
				}

				// クリア
				path.empty();

				// ファイルカウンター
				fileNum++;
			}

			// 後片付け
			FindClose(hFile);
		}
		else
		{
			// エラーメッセージ
			MessageBox(NULL, errorNS::SoundImportError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

#ifndef _DEBUG
			// 強制終了
			PostQuitMessage(0);
#endif
		}
	}
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
SoundBase::~SoundBase()
{
	// 連想配列の削除
	if (soundResource.size() > NULL)
	{
		// サウンドリソースの終了処理
		auto begin = soundResource.begin();
		auto end = soundResource.end();
		for (auto i = begin; i != end; i++)
		{
			// データ部の削除
			SAFE_DELETE_ARRAY(i->second.data)
		}
		soundResource.clear();
	}
}

//===================================================================================================================================
// テクスチャローダー
//===================================================================================================================================
bool SoundBase::LoadSound(const char *path, SoundResource *soundResource, int fileFormat)
{
	// ファイル
	FILE *fp = fopen(path, "rb");
	if (fp == NULL)
	{
		// エラーメッセージ
		std::string errorMsg = errorNS::SoundReadError;
		errorMsg += path;
		MessageBox(NULL, errorMsg.c_str(), MAIN_APP_NAME, (MB_OK | MB_ICONERROR));
		errorMsg.clear();

		return false;
	}

	if (fileFormat == FILE_FORMAT::FILE_WAV)
	{
		// WAVファイル
		ReadWav(fp, soundResource, path);
	}
	else if (fileFormat == FILE_FORMAT::FILE_OGG)
	{
		// Oggファイル
		if (!ReadOgg(soundResource, path))
		{
			return false;
		}
	}
	else
	{
		return false;
	}

	return true;
}

//===================================================================================================================================
// Oggの読み込み
//===================================================================================================================================
bool SoundBase::ReadOgg(SoundResource *soundResource, const char *path)
{
	// [Ogg]ファイルオープン
	OggVorbis_File oggVorbisFile = { NULL };
	if (ov_fopen(path, &oggVorbisFile) != S_OK)
	{
		return false;
	}

	// [Ogg]フォーマット情報
	vorbis_info* oggInfo = ov_info(&oggVorbisFile, -1);

	// [Ogg]データサイズ #16bit (ov_pcm_total(,-1) * チャンネル数 * sizeof(short))
	long oggAllSampling = static_cast<long>(ov_pcm_total(&oggVorbisFile, -1)) * oggInfo->channels;
	long oggAllSamplingSize = oggAllSampling * sizeof(short);
	soundResource->data = new short[oggAllSampling];
	int currentSection = 0;
	int comSize = 0;
	while (1)
	{
		// 一応4096にセットする(マニュアル通り)※09/21 後回し:例外があったら変更する
		char tmpBuffer[4096] = { NULL };
		// バイトオーダー-bigendianp:0(わざと1する人はいないと思うが…) 16bit-word:sizeof(short) 符号付き-sgned:sizeof(char)
		int readSize = ov_read(&oggVorbisFile, tmpBuffer, sizeof(tmpBuffer), 0, sizeof(short), 1, &currentSection);

		// デコードできるデータがない
		if (readSize == NULL)
		{
			break;
		}
		memcpy(soundResource->data + (comSize / sizeof(short)), tmpBuffer, readSize);
		comSize += readSize;
	}

	// サウンドリソースの作成
	soundResource->waveFormatEx.cbSize = 0;
	soundResource->waveFormatEx.nChannels = oggInfo->channels;
	soundResource->waveFormatEx.wBitsPerSample = 16;
	soundResource->waveFormatEx.nSamplesPerSec = oggInfo->rate;
	soundResource->waveFormatEx.wFormatTag = WAVE_FORMAT_PCM;
	soundResource->waveFormatEx.nBlockAlign = (oggInfo->channels*soundResource->waveFormatEx.wBitsPerSample) / 8;
	soundResource->waveFormatEx.nAvgBytesPerSec = oggInfo->rate*soundResource->waveFormatEx.nBlockAlign;
	soundResource->size = oggAllSamplingSize;
	soundResource->isWaveUpdate = true;
	soundResource->isCompressed = false;
	soundResource->isMix = false;
	//soundResource->data = new short[wavFile.data.size / sizeof(short)];
	//memcpy(soundResource->data, wavFile.data.data, wavFile.data.size);

	// [Ogg]後片付け
	ov_clear(&oggVorbisFile);

	return true;
}

//===================================================================================================================================
// WAVの読み込み
//===================================================================================================================================
void SoundBase::ReadWav(FILE *fp, SoundResource *soundResource, const char *path)
{
	// WAVファイル
	WAV_FILE wavFile = { NULL };

	// RIFFの読み込み
	if (!ReadRIFF(fp, &wavFile))
	{
		// エラーメッセージ
		std::string errorMsg = errorNS::SoundReadError;
		errorMsg += path;
		MessageBox(NULL, errorMsg.c_str(), MAIN_APP_NAME, (MB_OK | MB_ICONERROR));
		errorMsg.clear();
	}

	// チャンクループ
	int chunkSearchStatus = 0;
	while (chunkSearchStatus != soundBaseNS::chunkLoopEnd)
	{
		char	chunk[soundBaseNS::chunkSize];
		long	size = 0;

		// チャンクとサイズの読み込み
		fread(&chunk, sizeof(chunk), 1, fp);
		fread(&size, sizeof(size), 1, fp);

		// FMTの読み込み
		if (chunkSearchStatus != soundBaseNS::doneFmt)
		{
			if (ReadFMT(fp, &wavFile, chunk, size))
			{
				// 進捗
				chunkSearchStatus += soundBaseNS::doneFmt;
				continue;
			}
		}

		// DATAの読み込み
		if (chunkSearchStatus != soundBaseNS::dontData)
		{
			if (ReadDATA(fp, &wavFile, chunk, size))
			{
				// 進捗
				chunkSearchStatus += soundBaseNS::dontData;
				continue;
			}
		}

		// 他のチャンク
		fseek(fp, size, SEEK_CUR);
	}

	// サウンドリソースの作成
	soundResource->waveFormatEx.cbSize = 0;
	soundResource->waveFormatEx.nChannels = wavFile.fmt.channel;
	soundResource->waveFormatEx.wBitsPerSample = wavFile.fmt.bitPerSample;
	soundResource->waveFormatEx.nSamplesPerSec = wavFile.fmt.sampleRate;
	soundResource->waveFormatEx.wFormatTag = WAVE_FORMAT_PCM;
	soundResource->waveFormatEx.nBlockAlign = (wavFile.fmt.channel*wavFile.fmt.bitPerSample) / 8;
	soundResource->waveFormatEx.nAvgBytesPerSec = wavFile.fmt.sampleRate*soundResource->waveFormatEx.nBlockAlign;
	soundResource->size = wavFile.data.size;
	soundResource->data = new short[wavFile.data.size / sizeof(short)];
	memcpy(soundResource->data, wavFile.data.data, wavFile.data.size);
	soundResource->isWaveUpdate = true;
	soundResource->isCompressed = false;
	soundResource->isMix = false;

	// 後片付け
	SAFE_DELETE_ARRAY(wavFile.data.data)
	fclose(fp);
}

//===================================================================================================================================
// RIFFの読み込み
//===================================================================================================================================
bool SoundBase::ReadRIFF(FILE *fp, WAV_FILE *wavFile)
{
	fread(&wavFile->riff, sizeof(wavFile->riff), 1, fp);
	// ファイル検証
	if (memcmp(wavFile->riff.chunk, soundBaseNS::chunkRiff, sizeof(wavFile->riff.chunk)) ||
		(memcmp(wavFile->riff.waveChunk, soundBaseNS::chunkWave, sizeof(wavFile->riff.waveChunk))))
	{
		return false;
	}
	return true;
}

//===================================================================================================================================
// FMTの読み込み
//===================================================================================================================================
bool SoundBase::ReadFMT(FILE *fp, WAV_FILE *wavFile, char *chunk, long size)
{
	// FMTの読み込み
	if (memcmp(chunk, soundBaseNS::chunkFmt, soundBaseNS::chunkSize) == 0)
	{
		// チャンクとサイズの設定
		memcpy(wavFile->fmt.chunk, chunk, soundBaseNS::chunkSize);
		wavFile->fmt.size = size;

		// フォーマットIDから読み込み
		fread(&wavFile->fmt.formatTag, sizeof(FMT_CHUNK) - (soundBaseNS::chunkSize + sizeof(long)), 1, fp);

		// 成功
		return true;
	}
	return false;
}

//===================================================================================================================================
// DATAの読み込み
//===================================================================================================================================
bool SoundBase::ReadDATA(FILE *fp, WAV_FILE *wavFile, char *chunk, long size)
{
	// DATAの読み込み
	if (memcmp(chunk, soundBaseNS::chunkData, soundBaseNS::chunkSize) == 0)
	{
		// チャンクとサイズの設定
		memcpy(wavFile->data.chunk, chunk, soundBaseNS::chunkSize);
		wavFile->data.size = size;

		// メモリ確保
		wavFile->data.data = new short[wavFile->data.size / sizeof(short)];

		// 波形の読み込み
		fread(wavFile->data.data, wavFile->data.size, 1, fp);

		// 成功
		return true;
	}
	return false;
}