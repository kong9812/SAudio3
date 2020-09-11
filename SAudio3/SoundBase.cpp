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