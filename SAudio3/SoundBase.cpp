//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "SoundBase.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
SoundBase::SoundBase()
{
	WIN32_FIND_DATA FindFileData;	// �t�@�C���f�[�^
	int fileNum = 0;				// �t�@�C����

	for (int i = 0; i < FILE_FORMAT::FILE_MAX; i++)
	{
		// �t�@�C�������n���h��
		HANDLE hFile = FindFirstFileEx(soundBaseNS::fileFormat[i], FindExInfoBasic, &FindFileData,
			FindExSearchNameMatch, NULL, NULL);

		// �t�@�C������
		if (hFile != INVALID_HANDLE_VALUE)
		{
			// �t�@�C���p�X
			std::string path = "Sound\\";
			path += FindFileData.cFileName;

			// �V�F�[�_�[���\�[�X�̍쐬
			if (!LoadSound(path.c_str(), &soundResource[FindFileData.cFileName],i))
			{
				// �G���[���b�Z�[�W
				MessageBox(NULL, errorNS::SoundResourceError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

				// �����I��
				PostQuitMessage(0);
			}

			// �N���A
			path.empty();

			// �t�@�C���J�E���^�[
			fileNum++;

			while (FindNextFile(hFile, &FindFileData))
			{
				// �t�@�C���p�X
				path = "Sound\\";
				path += FindFileData.cFileName;

				// �V�F�[�_�[���\�[�X�̍쐬
				if (!LoadSound(path.c_str(), &soundResource[FindFileData.cFileName],i))
				{
					// �G���[���b�Z�[�W
					MessageBox(NULL, errorNS::SoundResourceError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

					// �����I��
					PostQuitMessage(0);
				}

				// �N���A
				path.empty();

				// �t�@�C���J�E���^�[
				fileNum++;
			}

			// ��Еt��
			FindClose(hFile);
		}
		else
		{
			// �G���[���b�Z�[�W
			MessageBox(NULL, errorNS::SoundImportError, MAIN_APP_NAME, (MB_OK | MB_ICONERROR));

#ifndef _DEBUG
			// �����I��
			PostQuitMessage(0);
#endif
		}
	}
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
SoundBase::~SoundBase()
{
	// �A�z�z��̍폜
	if (soundResource.size() > NULL)
	{
		// �T�E���h���\�[�X�̏I������
		auto begin = soundResource.begin();
		auto end = soundResource.end();
		for (auto i = begin; i != end; i++)
		{
			// �f�[�^���̍폜
			SAFE_DELETE_ARRAY(i->second.data)
		}
		soundResource.clear();
	}
}

#define TEST (false)
#if TEST
const short StepSize[89] = {
	7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
	19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
	50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
	130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
	337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
	876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
	2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
	5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
	15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
};

//***********************************************************************************************************************************
// �e�X�g
//***********************************************************************************************************************************
void test(short *data, long size, int channel)
{
	long newSize = 0;
	short lastData = 0;

	for (int i = 0; i < (size / (int)sizeof(short) / channel); i++)
	{
		for (int j = 0; j < channel; j++)
		{
			if (lastData >= data[i * channel + j])
			{
				// ����
				lastData = data[i * channel + j];
			}
			else
			{
				// ����
				lastData = data[i * channel + j];
			}
		}
	}
}

//***********************************************************************************************************************************
// �߂��l �񕪒T��
//***********************************************************************************************************************************
char FindADPCMID(short d)
{
	char result = 0;

	// �����ɕ�����
	if (d >= StepSize[44])
	{
		for (short i = 44; i <= 89; i++)
		{
			if (d == StepSize[i])
			{
				// ��������
				return i;
			}
			else if (d > StepSize[i])
			{
				return i - 1;
			}
		}
	}
	else
	{

	}
}
#endif

//===================================================================================================================================
// �����o��
//===================================================================================================================================
bool SoundBase::OutputSound(short *data, long size, int channel, long sampingPerSec, bool adpcm)
{
	// �t�@�C��
	FILE *fp = fopen("test.wav", "wb");

	WAV_FILE wavFile = { NULL };
	
	// RIFF
	memcpy(wavFile.riff.chunk, "RIFF", 4);
	wavFile.riff.size = 36 + size;
	memcpy(wavFile.riff.waveChunk, "WAVE", 4);
	fwrite(&wavFile.riff, sizeof(RIFF_CHUNK), 1, fp);

	// _fmt
	memcpy(wavFile.fmt.chunk, "fmt ", 4);
	wavFile.fmt.size = 16;
	wavFile.fmt.formatTag = 1;
	wavFile.fmt.channel = channel;
	wavFile.fmt.sampleRate = sampingPerSec;
	wavFile.fmt.bitPerSample = 16;
	wavFile.fmt.avgBytesPerSec = (wavFile.fmt.sampleRate * wavFile.fmt.bitPerSample) / 4;
	wavFile.fmt.blockAlign = wavFile.fmt.bitPerSample / 8;
	fwrite(&wavFile.fmt, sizeof(FMT_CHUNK), 1, fp);

	// data
	memcpy(wavFile.data.chunk, "data", 4);
	if (adpcm)
	{
#if TEST
		test(data, size, channel);
		wavFile.data.size = size;
		fwrite(&wavFile.data, sizeof(DATA_CHUNK) - 4, 1, fp);
		fwrite(data, size, 1, fp);
#else
		wavFile.data.size = size;
		fwrite(&wavFile.data, sizeof(DATA_CHUNK) - 4, 1, fp);
		fwrite(data, size, 1, fp);
#endif
	}
	else
	{
		wavFile.data.size = size;
		fwrite(&wavFile.data, sizeof(DATA_CHUNK) - 4, 1, fp);
		fwrite(data, size, 1, fp);
	}

	fclose(fp);

	return true;
}

//===================================================================================================================================
// �e�N�X�`�����[�_�[
//===================================================================================================================================
bool SoundBase::LoadSound(const char *path, SoundResource *soundResource, int fileFormat)
{
	// �t�@�C��
	FILE *fp = fopen(path, "rb");
	if (fp == NULL)
	{
		// �G���[���b�Z�[�W
		std::string errorMsg = errorNS::SoundReadError;
		errorMsg += path;
		MessageBox(NULL, errorMsg.c_str(), MAIN_APP_NAME, (MB_OK | MB_ICONERROR));
		errorMsg.clear();

		return false;
	}

	if (fileFormat == FILE_FORMAT::FILE_WAV)
	{
		// WAV�t�@�C��
		ReadWav(fp, soundResource, path);
	}
	else if (fileFormat == FILE_FORMAT::FILE_OGG)
	{
		// Ogg�t�@�C��
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
// Ogg�̓ǂݍ���
//===================================================================================================================================
bool SoundBase::ReadOgg(SoundResource *soundResource, const char *path)
{
	// [Ogg]�t�@�C���I�[�v��
	OggVorbis_File oggVorbisFile = { NULL };
	if (ov_fopen(path, &oggVorbisFile) != S_OK)
	{
		return false;
	}

	// [Ogg]�t�H�[�}�b�g���
	vorbis_info* oggInfo = ov_info(&oggVorbisFile, -1);

	// [Ogg]�f�[�^�T�C�Y #16bit (ov_pcm_total(,-1) * �`�����l���� * sizeof(short))
	long oggAllSampling = static_cast<long>(ov_pcm_total(&oggVorbisFile, -1)) * oggInfo->channels;
	long oggAllSamplingSize = oggAllSampling * sizeof(short);
	soundResource->data = new short[oggAllSampling];
	int currentSection = 0;
	int comSize = 0;
	while (1)
	{
		// �ꉞ4096�ɃZ�b�g����(�}�j���A���ʂ�)��09/21 ���:��O����������ύX����
		char tmpBuffer[4096] = { NULL };
		// �o�C�g�I�[�_�[-bigendianp:0(�킴��1����l�͂��Ȃ��Ǝv�����c) 16bit-word:sizeof(short) �����t��-sgned:sizeof(char)
		int readSize = ov_read(&oggVorbisFile, tmpBuffer, sizeof(tmpBuffer), 0, sizeof(short), 1, &currentSection);

		// �f�R�[�h�ł���f�[�^���Ȃ�
		if (readSize == NULL)
		{
			break;
		}
		memcpy(soundResource->data + (comSize / sizeof(short)), tmpBuffer, readSize);
		comSize += readSize;
	}

	// �T�E���h���\�[�X�̍쐬
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

	// [Ogg]��Еt��
	ov_clear(&oggVorbisFile);

	return true;
}

//===================================================================================================================================
// WAV�̓ǂݍ���
//===================================================================================================================================
void SoundBase::ReadWav(FILE *fp, SoundResource *soundResource, const char *path)
{
	// WAV�t�@�C��
	WAV_FILE wavFile = { NULL };

	// RIFF�̓ǂݍ���
	if (!ReadRIFF(fp, &wavFile))
	{
		// �G���[���b�Z�[�W
		std::string errorMsg = errorNS::SoundReadError;
		errorMsg += path;
		MessageBox(NULL, errorMsg.c_str(), MAIN_APP_NAME, (MB_OK | MB_ICONERROR));
		errorMsg.clear();
	}

	// �`�����N���[�v
	int chunkSearchStatus = 0;
	while (chunkSearchStatus != soundBaseNS::chunkLoopEnd)
	{
		char	chunk[soundBaseNS::chunkSize];
		long	size = 0;

		// �`�����N�ƃT�C�Y�̓ǂݍ���
		fread(&chunk, sizeof(chunk), 1, fp);
		fread(&size, sizeof(size), 1, fp);

		// FMT�̓ǂݍ���
		if (chunkSearchStatus != soundBaseNS::doneFmt)
		{
			if (ReadFMT(fp, &wavFile, chunk, size))
			{
				// �i��
				chunkSearchStatus += soundBaseNS::doneFmt;
				continue;
			}
		}

		// DATA�̓ǂݍ���
		if (chunkSearchStatus != soundBaseNS::dontData)
		{
			if (ReadDATA(fp, &wavFile, chunk, size))
			{
				// �i��
				chunkSearchStatus += soundBaseNS::dontData;
				continue;
			}
		}

		// ���̃`�����N
		fseek(fp, size, SEEK_CUR);
	}

	// �T�E���h���\�[�X�̍쐬
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

	// ��Еt��
	SAFE_DELETE_ARRAY(wavFile.data.data)
	fclose(fp);
}

//===================================================================================================================================
// RIFF�̓ǂݍ���
//===================================================================================================================================
bool SoundBase::ReadRIFF(FILE *fp, WAV_FILE *wavFile)
{
	fread(&wavFile->riff, sizeof(wavFile->riff), 1, fp);
	// �t�@�C������
	if (memcmp(wavFile->riff.chunk, soundBaseNS::chunkRiff, sizeof(wavFile->riff.chunk)) ||
		(memcmp(wavFile->riff.waveChunk, soundBaseNS::chunkWave, sizeof(wavFile->riff.waveChunk))))
	{
		return false;
	}
	return true;
}

//===================================================================================================================================
// FMT�̓ǂݍ���
//===================================================================================================================================
bool SoundBase::ReadFMT(FILE *fp, WAV_FILE *wavFile, char *chunk, long size)
{
	// FMT�̓ǂݍ���
	if (memcmp(chunk, soundBaseNS::chunkFmt, soundBaseNS::chunkSize) == 0)
	{
		// �`�����N�ƃT�C�Y�̐ݒ�
		memcpy(wavFile->fmt.chunk, chunk, soundBaseNS::chunkSize);
		wavFile->fmt.size = size;

		// �t�H�[�}�b�gID����ǂݍ���
		fread(&wavFile->fmt.formatTag, sizeof(FMT_CHUNK) - (soundBaseNS::chunkSize + sizeof(long)), 1, fp);

		// ����
		return true;
	}
	return false;
}

//===================================================================================================================================
// DATA�̓ǂݍ���
//===================================================================================================================================
bool SoundBase::ReadDATA(FILE *fp, WAV_FILE *wavFile, char *chunk, long size)
{
	// DATA�̓ǂݍ���
	if (memcmp(chunk, soundBaseNS::chunkData, soundBaseNS::chunkSize) == 0)
	{
		// �`�����N�ƃT�C�Y�̐ݒ�
		memcpy(wavFile->data.chunk, chunk, soundBaseNS::chunkSize);
		wavFile->data.size = size;

		// �������m��
		wavFile->data.data = new short[wavFile->data.size / sizeof(short)];

		// �g�`�̓ǂݍ���
		fread(wavFile->data.data, wavFile->data.size, 1, fp);

		// ����
		return true;
	}
	return false;
}