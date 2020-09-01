//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "SAudio3FadeXapo.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
SAudio3FadeXapo::SAudio3FadeXapo() :CXAPOParametersBase
(&(const XAPO_REGISTRATION_PROPERTIES)registrationProperties,
(BYTE *)xapoParameter, sizeof(SAudio3FadeParameter), false)
{
	// �G�t�F�N�^�[�̏��E����
	registrationProperties = {
	__uuidof(SAudio3FadeXapo),
	L"SAI_DELAY_APO",
	L"Copyright (C)2019 CHOI YAU KONG",
	1,
	0,
	XAPOBASE_DEFAULT_FLAG,
	1, 1, 1, 1 };
}

//===================================================================================================================================
// Process���O�̍ŏI����
// ���̓t�H�[�}�b�g�Əo�̓t�H�[�}�b�g�̐ݒ�
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
		memcpy(&inputFormat, pInputLockedParameters[0].pFormat, sizeof(inputFormat));
		memcpy(&outputFormat, pOutputLockedParameters[0].pFormat, sizeof(outputFormat));
	}

	return hr;
}

//===================================================================================================================================
// Process���O�̏������E�p�����[�^�[�̍ŏI�`�F�b�N
//===================================================================================================================================
void SAudio3FadeXapo::OnSetParameters
(const void* pParameters, UINT32 ParameterByteSize)
{
	SAudio3FadeParameter *tmpParameters = ((SAudio3FadeParameter *)pParameters);

	// �`�F�b�N���X�g
	XAPOASSERT(sizeof(SAudio3FadeParameter) > 0);
	XAPOASSERT(pParameters != NULL);
	XAPOASSERT(ParameterByteSize == sizeof(SAudio3FadeParameter));

	// ������

}

//===================================================================================================================================
// �G�t�F�N�g����
//===================================================================================================================================
void SAudio3FadeXapo::Process
(UINT32 inputProcessParameterCount,
	const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
	UINT32 outputProcessParameterCount,
	XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
	BOOL isEnabled)
{
	// �Đ��ʒu�̎擾

}