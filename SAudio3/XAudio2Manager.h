#pragma once
//===================================================================================================================================
// ���C�u�����t���O
//===================================================================================================================================
#define XAUDIO2_HELPER_FUNCTIONS

//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include <xaudio2.h>
#include "Main.h"
#include "SoundBase.h"
#include "XAudio2EffectManager.h"

//===================================================================================================================================
// �}�N����`
//===================================================================================================================================
#define SAFE_DESTROY_VOICE(p)			if(p){  (p)->DestroyVoice(); p = NULL; }

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
namespace xAudioManagerNS
{
	const float minVolume = 0.0f;
	const float maxVolume = 1.0f;
	const float overVolume = 10.0f;
}

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct VoiceResource
{
	bool isPlaying;							// �Đ����
	int effectCnt;							// �G�t�F�N�g�̌�
	std::list<XAPO_LIST> effectList;		// �G�t�F�N�g�̃��X�g
	IXAudio2SourceVoice *sourceVoice;
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class XAudio2Manager
{
public:
	XAudio2Manager(SoundBase *_soundBase);
	~XAudio2Manager();

	// �}�X�^�[�{�C�X�̍쐬
	IXAudio2MasteringVoice	*CreateMasterVoice(IXAudio2 *xAudio2);

	// �{�C�X���\�[�X�̍쐬
	HRESULT CreateVoiceResourceVoice(IXAudio2 *xAudio2, std::string voiceName, SoundResource soundResource);

	// �\�[�X�{�C�X�̍Đ��E�ꎞ��~
	void PlayPauseSourceVoice(IXAudio2 *xAudio2, std::string voiceName);

	// �}�X�^�[�{�C�X�{�����[���̎擾
	float GetMasteringVoiceVolumeLevel(void);
	// �}�X�^�[�{�C�X�{�����[���̒���
	HRESULT SetMasteringVoiceVolumeLevel(float);

	// �Đ����
	bool GetIsPlaying(std::string voiceName);

	// �{�C�X���
	XAUDIO2_VOICE_STATE GetVoiceState(std::string voiceName);

	// �{�C�X�ڍ�
	XAUDIO2_VOICE_DETAILS GetVoiceDetails(std::string voiceName);

	// �����T���v�����O�̐ݒ�
	void SetProcessSampling(int _processSampling);

	// �����T���v�����O�̎擾
	int GetProcessSampling(void);

	// �T�u�~�b�N�X�{�C�X�̍쐬
	IXAudio2SubmixVoice *CreateSubmixVoice(std::string voiceName);
	
	// �T�u�~�b�N�X�{�C�X�̍쐬
	IXAudio2SourceVoice *CreateSourceVoice(std::string voiceName);

	// �A�E�g�v�b�g�{�C�X�̐ݒ�
	HRESULT SetOutputVoice(std::string voiceName,
		std::map <std::string, XAUDIO2_SEND_DESCRIPTOR> sendDescriptorList, int sendCount);

private:
	SoundBase				*soundBase;							// �T�E���h�x�[�X

	IXAudio2				*XAudio2;							// XAudio2
	IXAudio2MasteringVoice	*XAudio2MasteringVoice;				// �}�X�^�[�{�C�X
	XAudio2EffectManager	*xAudio2EffectManager;				// �G�t�F�N�g�}�l�[�W���[
	std::map<std::string, VoiceResource> voiceResource;			// �{�C�X���\�[�X

	int						processSampling;					// �����T���v�����O
	float					oldMasteringVoiceVolume;			// �d���΍�
};