//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#pragma once
#include "Main.h"
#include <xaudio2.h>

//===================================================================================================================================
// �萔��`
//===================================================================================================================================

//===================================================================================================================================
// �}�N����`
//===================================================================================================================================
#define SAFE_DESTROY_VOICE(p)			if(p){  (p)->DestroyVoice(); p = NULL; }

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class XAudio2Manager
{
public:
	XAudio2Manager();
	~XAudio2Manager();

	// �}�X�^�[�{�C�X�̍쐬
	IXAudio2MasteringVoice *CreateMasterVoice(IXAudio2 *xAudio2);

private:
	IXAudio2 *XAudio2;								// XAudio2
	IXAudio2MasteringVoice *XAudio2MasteringVoice;	// �}�X�^�[�{�C�X
};