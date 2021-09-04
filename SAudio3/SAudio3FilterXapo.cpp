//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "SAudio3FilterXapo.h"

//===================================================================================================================================
// �G�t�F�N�^�[�̏��E����
//===================================================================================================================================
XAPO_REGISTRATION_PROPERTIES SAudio3FilterXapo::registrationProperties = {
	__uuidof(SAudio3FilterXapo),
	L"SAudio3FilterXapo",
	L"Copyright (C)2021 CHOI YAU KONG",
	1,
	1,
	XAPOBASE_DEFAULT_FLAG,
	1, 1, 1, 1 };

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
SAudio3FilterXapo::SAudio3FilterXapo() :CXAPOParametersBase(&registrationProperties,
(BYTE *)xapoParameter, sizeof(SAudio3FilterParameter), false)
{
	alpha = NULL;
	omega = NULL;
	memset(inputBackup, NULL, sizeof(inputBackup));
	memset(outputBackup, NULL, sizeof(outputBackup));
	memset(a, NULL, sizeof(a));
	memset(b, NULL, sizeof(b));
}

//===================================================================================================================================
// �@���̓t�H�[�}�b�g�Əo�̓t�H�[�}�b�g�̐ݒ�
//===================================================================================================================================
HRESULT SAudio3FilterXapo::LockForProcess
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
void SAudio3FilterXapo::SetParameters
(const void* pParameters, UINT32 ParameterByteSize)
{
	// �T�C�Y�`�F�b�N�̂�
	if (ParameterByteSize == sizeof(SAudio3FilterParameter))
	{
		return CXAPOParametersBase::SetParameters(pParameters, ParameterByteSize);
	}

	return;
}

//===================================================================================================================================
// �BProcess���O�̏������E�p�����[�^�[�̍ŏI�`�F�b�N
//===================================================================================================================================
void SAudio3FilterXapo::OnSetParameters
(const void* pParameters, UINT32 ParameterByteSize)
{
	SAudio3FilterParameter *tmpParameters = ((SAudio3FilterParameter *)pParameters);

	// �`�F�b�N���X�g
	XAPOASSERT(sizeof(SAudio3FilterParameter) > 0);
	XAPOASSERT(pParameters != NULL);
	XAPOASSERT(ParameterByteSize == sizeof(SAudio3FilterParameter));
	XAPOASSERT(tmpParameters->type < XAPO_FILTER_TYPE::XFT_MAX);
	XAPOASSERT(tmpParameters->Q <= 5);
	XAPOASSERT(tmpParameters->cutoffFreq <= (inputFormat.nSamplesPerSec / 2));
	XAPOASSERT((tmpParameters->bandwidth <= 5) && (tmpParameters->bandwidth > 0));

	// �t�B���^�W���̌v�Z
	switch (tmpParameters->type)
	{
	case XAPO_FILTER_TYPE::XFT_LowpassFilter:
	{
		omega = HS_TO_RADIAN_FREQUENCY(tmpParameters->cutoffFreq, inputFormat.nSamplesPerSec);
		alpha = sinf(omega) / (2.0f * tmpParameters->Q);
		a[0] = 1.0f + alpha;
		a[1] = -2.0f * cosf(omega);
		a[2] = 1.0f - alpha;
		b[0] = (1.0f - cosf(omega)) / 2.0f;
		b[1] = 1.0f - cosf(omega);
		b[2] = (1.0f - cosf(omega)) / 2.0f;
	}
	break;
	case XAPO_FILTER_TYPE::XFT_HighpassFilter:
	{
		omega = HS_TO_RADIAN_FREQUENCY(tmpParameters->cutoffFreq, inputFormat.nSamplesPerSec);
		alpha = sinf(omega) / (2.0f * tmpParameters->Q);
		a[0] = 1.0f + alpha;
		a[1] = -2.0f * cosf(omega);
		a[2] = 1.0f - alpha;
		b[0] = (1.0f + cosf(omega)) / 2.0f;
		b[1] = -(1.0f + cosf(omega));
		b[2] = (1.0f + cosf(omega)) / 2.0f;
	}
	break;
	case XAPO_FILTER_TYPE::XFT_BandpassFilter:
	{
		omega = HS_TO_RADIAN_FREQUENCY(tmpParameters->cutoffFreq, inputFormat.nSamplesPerSec);
		alpha = sinf(omega) * sinf(logf(2.0f) / 2.0 * tmpParameters->bandwidth * omega / sinf(omega));

		a[0] = 1.0f + alpha;
		a[1] = -2.0f * cosf(omega);
		a[2] = 1.0f - alpha;
		b[0] = alpha;
		b[1] = 0.0f;
		b[2] = -alpha;
	}
	break;
	case XAPO_FILTER_TYPE::XFT_NotchFilter:
	{
		omega = HS_TO_RADIAN_FREQUENCY(tmpParameters->cutoffFreq, inputFormat.nSamplesPerSec);
		alpha = sinf(omega) * sinf(logf(2.0f) / 2.0 * tmpParameters->bandwidth * omega / sinf(omega));

		a[0] = 1.0f + alpha;
		a[1] = -2.0f * cosf(omega);
		a[2] = 1.0f - alpha;
		b[0] = 1.0f;
		b[1] = -2.0f * cosf(omega);
		b[2] = 1.0f;
	}
	break;
	default:
	{
		// ���[�p�X
		omega = HS_TO_RADIAN_FREQUENCY(tmpParameters->cutoffFreq, inputFormat.nSamplesPerSec);
		alpha = sinf(omega) / (2.0f * tmpParameters->Q);
		a[0] = 1.0f + alpha;
		a[1] = -2.0f * cosf(omega);
		a[2] = 1.0f - alpha;
		b[0] = (1.0f - cosf(omega)) / 2.0f;
		b[1] = 1.0f - cosf(omega);
		b[2] = (1.0f - cosf(omega)) / 2.0f;
	}
	break;
	}

	memset(inputBackup, NULL, sizeof(inputBackup));
	memset(outputBackup, NULL, sizeof(outputBackup));

	return;
}

//===================================================================================================================================
// �C�G�t�F�N�g����
//===================================================================================================================================
void SAudio3FilterXapo::Process
(UINT32 inputProcessParameterCount,
	const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
	UINT32 outputProcessParameterCount,
	XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
	BOOL isEnabled)
{
	// ���̃p�����[�^�\���� = �g�p����p�����[�^�̃|�C�����[
	SAudio3FilterParameter *tmpParameter = (SAudio3FilterParameter *)BeginProcess();

	if (isEnabled)
	{
		if (pInputProcessParameters->BufferFlags != XAPO_BUFFER_FLAGS::XAPO_BUFFER_SILENT)
		{
			for (int i = 0; i < ((int)pInputProcessParameters->ValidFrameCount * inputFormat.nChannels); i++)
			{
				// ���o�͂̃o�b�t�@
				float *inputBuf = (float *)pInputProcessParameters->pBuffer;
				float *outputBuf = (float *)pOutputProcessParameters->pBuffer;

				// TEST(�������[�p�X)
				outputBuf[i] = BiQuadFilter(inputBuf[i]);
			}
		}
	}

	// �G���h�v���Z�X
	EndProcess();
}

//===================================================================================================================================
// �C�G�t�F�N�g����
//===================================================================================================================================
float SAudio3FilterXapo::BiQuadFilter(float input)
{
	// ���͐M���Ƀt�B���^��K�p���A�o�͐M���Ƃ��ď����o���B
	float output = b[0] / a[0] * input + b[1] / a[0] * inputBackup[0] + b[2] / a[0] * inputBackup[1]
		- a[1] / a[0] * outputBackup[0] - a[2] / a[0] * outputBackup[1];

	inputBackup[1] = inputBackup[0];		// 2�O�̓��͐M�����X�V
	inputBackup[0] = input;					// 1�O�̓��͐M�����X�V

	outputBackup[1] = outputBackup[0];      // 2�O�̏o�͐M�����X�V
	outputBackup[0] = output;				// 1�O�̏o�͐M�����X�V

	return output;
}
