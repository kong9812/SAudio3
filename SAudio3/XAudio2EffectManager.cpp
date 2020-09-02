//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "XAudio2EffectManager.h"
#include "SAudio3FadeXapo.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
XAudio2EffectManager::XAudio2EffectManager()
{
	XApo = new IUnknown*[XAPO_LIST::XAPO_MAX];
	XApo[XAPO_LIST::XAPO_FADE] = (IXAPO *)new SAudio3FadeXapo();
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
XAudio2EffectManager::~XAudio2EffectManager()
{
	// XAPO�̏I������
	for (int i = 0; i < XAPO_LIST::XAPO_MAX; i++)
	{
		SAFE_RELEASE(XApo[i])
	}
	SAFE_DELETE(XApo)
}

//===================================================================================================================================
// �G�t�F�N�g�̐ݒu�E����
//===================================================================================================================================
HRESULT XAudio2EffectManager::SetXapoEffect(IXAudio2SubmixVoice *submixVoice, XAPO_LIST xapoID,
	int effectCnt, std::list<XAPO_LIST> effectList, bool isUse)
{
	bool isInit = false;	// ���������
	int effectID = NULL;	// �G�t�F�N�gID
	if (effectList.size() > NULL)
	{
		// �G�t�F�N�g���X�g����
		for (auto _effectID : effectList)
		{
			// �Y������G�t�F�N�g�����݂�����
			if (_effectID == xapoID)
			{
				isInit = true;			// ���ɏ���������Ă���
				effectID = _effectID;	// �R�s�[
			}
		}
	}
	if (isUse)
	{
		// �G�t�F�N�g����������Ă��Ȃ�
		if (!isInit)
		{
			XAUDIO2_EFFECT_DESCRIPTOR	effectDescriptor = { NULL };	// �G�t�F�N�g�f�B�X�N���v�^
			XAUDIO2_EFFECT_CHAIN		chain = { NULL };				// �G�t�F�N�g�`�F��

			XAUDIO2_VOICE_DETAILS		voiceDetails = { NULL };		// �{�C�X�ڍ�
			submixVoice->GetVoiceDetails(&voiceDetails);				// �{�C�X�ڍׂ̎擾

			// �G�t�F�N�g�f�B�X�N���v�^�̏�����
			effectDescriptor.pEffect = XApo[xapoID];
			effectDescriptor.InitialState = isUse;
			effectDescriptor.OutputChannels = voiceDetails.InputChannels;

			// �G�t�F�N�g�`�F���̏�����
			chain.EffectCount = effectCnt;
			chain.pEffectDescriptors = &effectDescriptor;

			// �\�[�X�{�C�X�Ƃ̐ڑ�
			submixVoice->SetEffectChain(&chain);
		}
		else
		{
			// �G�t�F�N�g�̗L����
			submixVoice->EnableEffect(effectID);
		}
	}
	else
	{
		// �G�t�F�N�g�̖�����
		submixVoice->DisableEffect(effectID);
	}

	return S_OK;
}