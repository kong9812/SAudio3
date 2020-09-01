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
	bool isPlaying;
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
	void CreateVoiceResourceVoice(IXAudio2 *xAudio2, std::string voiceName, SoundResource soundResource);

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

private:
	SoundBase				*soundBase;							// �T�E���h�x�[�X

	IXAudio2				*XAudio2;							// XAudio2
	IXAudio2MasteringVoice	*XAudio2MasteringVoice;				// �}�X�^�[�{�C�X
	XAudio2EffectManager	*xAudio2EffectManager;				// �G�t�F�N�g�}�l�[�W���[
	std::map<std::string, VoiceResource> voiceResource;			// �{�C�X���\�[�X

	float					oldMasteringVoiceVolume;			// �d���΍�
};