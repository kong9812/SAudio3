#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "TextureBase.h"
#include "SoundBase.h"

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct Mixer_Resource
{
	std::string soundName;	// �T�E���h��
	int cnt;				// ���p��
};

struct Mixer_Parameter
{
	std::string soundName;
	bool	isFade;
	bool	isPlaying;
	float	playingPos;
	int		fadeInPos;
	int		fadeOutPos;
	int		fadeInMs;
	int		fadeOutMs;
};

struct Mixer_Data
{
	std::list<Mixer_Resource> mixerResource;	// �~�N�T�[���\�[�X
	std::list<Mixer_Parameter> mixerParameter;		// �~�N�T�[�p�����[�^�[
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class ImGuiMixerManager
{
public:
	ImGuiMixerManager(TextureBase *_textureBase);
	~ImGuiMixerManager();

	// ���p����T�E���h�̐ݒu
	void SetMixerResource(std::string soundName, bool addUse);

	// �~�N�T�[�p�l��
	void MixerPanel(bool *showMixerPanael);

private:
	Mixer_Data	mixerData;		// �~�N�T�[�f�[�^
	TextureBase	*textureBase;	// �e�N�X�`���x�[�X

	// �~�N�T�[�p�����[�^�[�̍쐬
	Mixer_Parameter CreateMixerParameter(Mixer_Resource mixResourceData);

	// [�p�[�c]�Đ��v���C���[
	void MixerPartPlayer(std::list<Mixer_Parameter>::iterator mixerParameter, float buttomSize);

	// [�p�[�c]�폜�{�^��
	void MixerPartDelete(bool deleteButton);

	// [�p�[�c]�~�N�T�[
	void MixerPartMixer(std::list<Mixer_Parameter>::iterator mixerParameter);
};