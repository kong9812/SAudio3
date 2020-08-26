#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
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
}

//===================================================================================================================================
// �\����
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
	bool isWaveUpdate;		// �g�`�̍ĕ`�悪�K�v?(�����l:true)
	bool isCompressed;		// ���k����?(�����l:false)
	long size;
	short *data;
	WAVEFORMATEX waveFormatEx;
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class SoundBase
{
public:
	SoundBase();
	~SoundBase();
	
	std::map<std::string, SoundResource> soundResource;	// [�A�z�z��]�T�E���h���\�[�X

private:

	// �e�N�X�`�����[�_�[
	bool LoadSound(const char *path, SoundResource *soundResource);

	// RIFF�̓ǂݍ���
	bool ReadRIFF(FILE *fp, WAV_FILE *wavFile);

	// FMT�̓ǂݍ���
	bool ReadFMT(FILE *fp, WAV_FILE *wavFile, char *chunk, long size);

	// DATA�̓ǂݍ���
	bool ReadDATA(FILE *fp, WAV_FILE *wavFile, char *chunk, long size);
};