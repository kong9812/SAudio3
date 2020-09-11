#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include <xapo.h>
#include <xapobase.h>
#include <xapofx.h>
#include <xaudio2.h>
#pragma comment(lib,"xapobase.lib")

#include "Main.h"

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
enum XAPO_LIST
{
	XAPO_FADE,
	XAPO_MAX
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class XAudio2EffectManager
{
public:
	XAudio2EffectManager();
	~XAudio2EffectManager();

	// �t�F�[�h�̐ݒu
	HRESULT SetXapoFade(IXAudio2SourceVoice *sourceVoice);

	// �G�t�F�N�g�̐ݒu�E����
	HRESULT SetXapoEffect(IXAudio2SourceVoice *sourceVoice, XAPO_LIST xapoID,
		int effectCnt, std::list<XAPO_LIST> effectList, bool isUse);

private:
};