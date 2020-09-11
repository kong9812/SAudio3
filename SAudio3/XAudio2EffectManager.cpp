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
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
XAudio2EffectManager::~XAudio2EffectManager()
{
	//// XAPO�̏I������
	//SAFE_RELEASE(XApo)
}

//===================================================================================================================================
// �t�F�[�h�̐ݒu
//===================================================================================================================================
HRESULT XAudio2EffectManager::SetXapoFade(IXAudio2SourceVoice *sourceVoice)
{
	XAUDIO2_EFFECT_DESCRIPTOR	effectDescriptor = { NULL };	// �G�t�F�N�g�f�B�X�N���v�^
	XAUDIO2_EFFECT_CHAIN		chain = { NULL };				// �G�t�F�N�g�`�F��

	XAUDIO2_VOICE_DETAILS		voiceDetails = { NULL };		// �{�C�X�ڍ�
	sourceVoice->GetVoiceDetails(&voiceDetails);				// �{�C�X�ڍׂ̎擾

	// XAPOs
	IUnknown *XApo = (IXAPO *)new SAudio3FadeXapo();
	// �G�t�F�N�g�f�B�X�N���v�^�̏�����
	effectDescriptor.pEffect = XApo;
	effectDescriptor.InitialState = true;
	effectDescriptor.OutputChannels = voiceDetails.InputChannels;

	// �G�t�F�N�g�`�F���̏�����
	chain.EffectCount = 1;
	chain.pEffectDescriptors = &effectDescriptor;

	// �\�[�X�{�C�X�Ƃ̐ڑ�
	sourceVoice->SetEffectChain(&chain);

	// �����ۂ��I(���Ԃ���v�c�m�F�҂�)
	SAFE_RELEASE(XApo);

	return S_OK;
}

//===================================================================================================================================
// �G�t�F�N�g�̐ݒu�E����
//===================================================================================================================================
HRESULT XAudio2EffectManager::SetXapoEffect(IXAudio2SourceVoice *sourceVoice, XAPO_LIST xapoID,
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
			sourceVoice->GetVoiceDetails(&voiceDetails);				// �{�C�X�ڍׂ̎擾

			// �G�t�F�N�g�f�B�X�N���v�^�̏�����
			IUnknown *XApo = (IXAPO *)new SAudio3FadeXapo();;
			effectDescriptor.pEffect = XApo;
			effectDescriptor.InitialState = isUse;
			effectDescriptor.OutputChannels = voiceDetails.InputChannels;

			// �G�t�F�N�g�`�F���̏�����
			chain.EffectCount = effectCnt;
			chain.pEffectDescriptors = &effectDescriptor;

			// �\�[�X�{�C�X�Ƃ̐ڑ�
			sourceVoice->SetEffectChain(&chain);
		}
		else
		{
			// �G�t�F�N�g�̗L����
			sourceVoice->EnableEffect(effectID);
		}
	}
	else
	{
		// �G�t�F�N�g�̖�����
		sourceVoice->DisableEffect(effectID);
	}

	return S_OK;
}