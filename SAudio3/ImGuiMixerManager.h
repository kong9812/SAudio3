#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "TextureBase.h"
#include "SoundBase.h"
#include "SAudio3FadeXapo.h"

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct Mixer_Resource
{
	//std::map <std::string, XAUDIO2_SEND_DESCRIPTOR>sendDescriptor;
	std::string soundName;	// �T�E���h��
	int cnt;				// ���p��
};

struct Mixer_Parameter
{
	SAudio3FadeParameter sAudio3FadeParameter;		
	IXAudio2SourceVoice *XAudio2SourceVoice;	// �e�X�g�Đ��p
	std::string soundName;
	std::string parameterName;
	bool	isFade;
	bool	isPlaying;
	float	playingPos;
	int		maxSample;
	int		maxMs;
};

struct Mixer_Data
{
	std::list<Mixer_Resource> mixerResource;	// �~�N�T�[���\�[�X
	std::list<Mixer_Parameter> mixerParameter;	// �~�N�T�[�p�����[�^�[
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class ImGuiMixerManager
{
public:
	ImGuiMixerManager(XAudio2Manager *_xAudio2Manager, TextureBase *_textureBase, SoundBase *_soundBase);
	~ImGuiMixerManager();

	// ���p����T�E���h�̐ݒu
	void SetMixerResource(std::string soundName, bool addUse);

	// �~�N�T�[�p�l��
	void MixerPanel(bool *showMixerPanael);

private:
	XAudio2Manager	*xAudio2Manager;// XAudio2�}�l�W���[
	TextureBase		*textureBase;	// �e�N�X�`���x�[�X
	SoundBase		*soundBase;		// �T�E���h�x�[�X
	Mixer_Data		mixerData;		// �~�N�T�[�f�[�^

	// �~�N�T�[�p�����[�^�[�̍쐬
	Mixer_Parameter CreateMixerParameter(Mixer_Resource mixResourceData);

	// [�p�[�c]�Đ��v���C���[
	void MixerPartPlayer(std::list<Mixer_Parameter>::iterator mixerParameter, float buttomSize);

	// [�p�[�c]�폜�{�^��
	bool MixerPartDelete(std::list<Mixer_Parameter>::iterator mixerParameter, bool deleteButton);

	// [�p�[�c]�~�N�T�[
	void MixerPartMixer(std::list<Mixer_Parameter>::iterator mixerParameter);

	//// ���M�f�B�X�N���v�^�̍쐬�E�ݒu
	//void SetSendDescriptor(std::string mixerParameterName,
	//	std::list<Mixer_Resource>::iterator mixerResource, IXAudio2SubmixVoice *XAudio2SubmixVoice);
};