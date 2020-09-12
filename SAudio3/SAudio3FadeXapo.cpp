//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "SAudio3FadeXapo.h"

//===================================================================================================================================
// �G�t�F�N�^�[�̏��E����
//===================================================================================================================================
XAPO_REGISTRATION_PROPERTIES SAudio3FadeXapo::registrationProperties = {
	__uuidof(SAudio3FadeXapo),
	L"SAudio3FadeXapo",
	L"Copyright (C)2020 CHOI YAU KONG",
	1,
	1,
	XAPOBASE_DEFAULT_FLAG,
	1, 1, 1, 1 };

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
SAudio3FadeXapo::SAudio3FadeXapo() :CXAPOParametersBase(&registrationProperties,
(BYTE *)xapoParameter, sizeof(SAudio3FadeParameter), false)
{
	samplingCnt = NULL;		// �T���v�����O�J�E���^�[(�����ʒu)
	fadeAddVolume = NULL;	// [�t�F�C�h�C��]1�T���v�����O������̃{�����[��
	fadeMinusVolume = NULL;	// [�t�F�C�h�A�E�g]1�T���v�����O������̃{�����[��
}

//===================================================================================================================================
// �@���̓t�H�[�}�b�g�Əo�̓t�H�[�}�b�g�̐ݒ�
//===================================================================================================================================
HRESULT SAudio3FadeXapo::LockForProcess
(UINT32 inputLockedParameterCount,
	const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pInputLockedParameters,
	UINT32 outputLockedParameterCount,
	const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pOutputLockedParameters)
{
	const HRESULT hr = CXAPOParametersBase::LockForProcess(
		inputLockedParameterCount,
		pInputLockedParameters,
		outputLockedParameterCount,
		pOutputLockedParameters);

	if (SUCCEEDED(hr))
	{
		// 0�Ԗڂ�����o��
		memcpy(&inputFormat, pInputLockedParameters[0].pFormat, sizeof(inputFormat));
		memcpy(&outputFormat, pOutputLockedParameters[0].pFormat, sizeof(outputFormat));
	}

	return hr;
}

//===================================================================================================================================
// �A�p�����[�^�[�`�F�b�N
//===================================================================================================================================
void SAudio3FadeXapo::SetParameters
(const void* pParameters, UINT32 ParameterByteSize)
{
	// �T�C�Y�`�F�b�N�̂�
	if (ParameterByteSize == sizeof(SAudio3FadeParameter))
	{
		return CXAPOParametersBase::SetParameters(pParameters, ParameterByteSize);
	}

	return;
}

//===================================================================================================================================
// �BProcess���O�̏������E�p�����[�^�[�̍ŏI�`�F�b�N
//===================================================================================================================================
void SAudio3FadeXapo::OnSetParameters
(const void* pParameters, UINT32 ParameterByteSize)
{
	SAudio3FadeParameter *tmpParameters = ((SAudio3FadeParameter *)pParameters);

	// �`�F�b�N���X�g
	XAPOASSERT(sizeof(SAudio3FadeParameter) > 0);
	XAPOASSERT(pParameters != NULL);
	XAPOASSERT(ParameterByteSize == sizeof(SAudio3FadeParameter));
	XAPOASSERT(tmpParameters->fadeInStartMs != tmpParameters->fadeInEndMs);
	XAPOASSERT(tmpParameters->fadeOutStartMs != tmpParameters->fadeOutEndMs);
	XAPOASSERT(tmpParameters->fadeInStartMs < tmpParameters->fadeOutStartMs);
	XAPOASSERT(tmpParameters->fadeInStartMs < tmpParameters->fadeOutEndMs);
	XAPOASSERT(tmpParameters->fadeInEndMs < tmpParameters->fadeOutStartMs);
	XAPOASSERT(tmpParameters->fadeInEndMs < tmpParameters->fadeOutEndMs);

	// �t�F�C�h�C��
	fadeInStartSampling = MS_TO_SAMPLING(tmpParameters->fadeInStartMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeInEndSampling = MS_TO_SAMPLING(tmpParameters->fadeInEndMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeAddVolume = 1.0f / ((float)fadeInEndSampling - (float)fadeInStartSampling);
	// �t�F�C�h�A�E�g
	fadeOutStartSampling = MS_TO_SAMPLING(tmpParameters->fadeOutStartMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeOutEndSampling = MS_TO_SAMPLING(tmpParameters->fadeOutEndMs, inputFormat.nSamplesPerSec*inputFormat.nChannels);
	fadeMinusVolume = 1.0f / ((float)fadeOutEndSampling - (float)fadeOutStartSampling);
}

//===================================================================================================================================
// �C�G�t�F�N�g����
//===================================================================================================================================
void SAudio3FadeXapo::Process
(UINT32 inputProcessParameterCount,
	const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
	UINT32 outputProcessParameterCount,
	XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
	BOOL isEnabled)
{
	// ���̃p�����[�^�\���� = �g�p����p�����[�^�̃|�C�����[
	SAudio3FadeParameter *tmpParameter = (SAudio3FadeParameter *)BeginProcess();

	if (isEnabled)
	{
		if (pInputProcessParameters->BufferFlags != XAPO_BUFFER_FLAGS::XAPO_BUFFER_SILENT)
		{
			for (int i = 0; i < ((int)pInputProcessParameters->ValidFrameCount * inputFormat.nChannels); i++)
			{
				// ���o�͂̃o�b�t�@
				float *inputBuf = (float *)pInputProcessParameters->pBuffer;
				float *outputBuf = (float *)pOutputProcessParameters->pBuffer;

				// �t�F�C�h�C������
				if ((samplingCnt >= fadeInStartSampling) &&
					(samplingCnt <= fadeInEndSampling))
				{
					// �t�F�C�h���̈ʒu
					int fadeIdx = samplingCnt - fadeInStartSampling;

					// �{�����[���v�Z
					float volume = (fadeAddVolume)*fadeIdx;
					outputBuf[i] = inputBuf[i] * volume;
				}
				else if ((samplingCnt >= fadeOutStartSampling) &&
						(samplingCnt <= fadeOutEndSampling))
				{
					// �t�F�C�h���̈ʒu
					int fadeIdx = samplingCnt - fadeOutStartSampling;

					// �{�����[���v�Z
					float volume = 1.0f - ((fadeMinusVolume)*fadeIdx);
					outputBuf[i] = inputBuf[i] * volume;
				}

				// �T���v�����O�J�E���^�[(�����ʒu)
				samplingCnt++;
				if (samplingCnt >= tmpParameter->allSampling)
				{
					samplingCnt = 0;
				}
			}
		}
	}
	// �G���h�v���Z�X
	EndProcess();
}

//===================================================================================================================================
// �����M�p
//===================================================================================================================================
void SAudio3FadeXapo::GetParameters
(void* pParameters, UINT32 ParameterByteSize)
{
	// �����i��
	if (ParameterByteSize == sizeof(int))
	{
		*(int *)pParameters = samplingCnt;
	}
}