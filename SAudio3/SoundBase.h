#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"

//===================================================================================================================================
// 定数定義
//===================================================================================================================================
enum FILE_FORMAT
{
	FILE_WAV,
	FILE_OGG,
	FILE_MAX
};

namespace soundBaseNS
{
	const int chunkSize		= 4;
	const int chunkLoopEnd	= 0b11;
	const int doneFmt		= 0b1;
	const int dontData		= 0b1 << 0b1;
	const char * const chunkRiff	= "RIFF";
	const char * const chunkWave	= "WAVE";
	const char * const chunkFmt		= "fmt ";
	const char * const chunkData	= "data";

	const char * const fileFormat[FILE_FORMAT::FILE_MAX] = { "Sound\\*.wav*","Sound\\*.ogg*" };
}

//===================================================================================================================================
// 構造体
//===================================================================================================================================
struct RIFF_CHUNK
{
	char	chunk[4];
	long	size;
	char	waveChunk[4];
};

struct FMT_CHUNK
{
	char	chunk[4];
	long	size;
	short	formatTag;
	short	channel;
	long	sampleRate;
	long	avgBytesPerSec;
	short	blockAlign;
	short	bitPerSample;
};

struct DATA_CHUNK
{
	char	chunk[4];
	long	size;
	short	*data;
};

struct WAV_FILE
{
	RIFF_CHUNK	riff;
	FMT_CHUNK	fmt;
	DATA_CHUNK	data;
};

struct SoundResource
{
	bool isWaveUpdate;		// 波形の再描画が必要?(初期値:true)
	bool isCompressed;		// 圧縮完了?(初期値:false)
	bool isMix;				// ミックスに使う?(初期値:false)
	long size;
	short *data;
	WAVEFORMATEX waveFormatEx;
};

//===================================================================================================================================
// クラス
//===================================================================================================================================
class SoundBase
{
public:
	SoundBase();
	~SoundBase();
	
	std::map<std::string, SoundResource> soundResource;	// [連想配列]サウンドリソース

private:

	// テクスチャローダー
	bool LoadSound(const char *path, SoundResource *soundResource, int fileFormat);

	// Oggの読み込み
	bool ReadOgg(SoundResource *soundResource, const char *path);

	// WAVの読み込み
	void ReadWav(FILE *fp, SoundResource *soundResource, const char *path);

	// RIFFの読み込み
	bool ReadRIFF(FILE *fp, WAV_FILE *wavFile);

	// FMTの読み込み
	bool ReadFMT(FILE *fp, WAV_FILE *wavFile, char *chunk, long size);

	// DATAの読み込み
	bool ReadDATA(FILE *fp, WAV_FILE *wavFile, char *chunk, long size);
};