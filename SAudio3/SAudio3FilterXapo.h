#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// �}�N����`
//===================================================================================================================================
#define HS_TO_RADIAN_FREQUENCY(hs,sf)	(2.0f * M_PI * hs / sf)
#define S_HS_TO_RADIAN_FREQUENCY(hs,sf)	(hs / sf * 6.f)				// �ȈՉ�

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
enum XAPO_FILTER_TYPE : int
{
	XFT_LowpassFilter,
	XFT_HighpassFilter,
	XFT_BandpassFilter,
	XFT_NotchFilter,
	XFT_MAX
};

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct SAudio3FilterParameter
{
	XAPO_FILTER_TYPE type;		// �t�B���^�[�̎��
	int cutoffFreq;				// �J�b�g�I�t���g��
	float Q;					// Q�l
	float bandwidth;			// �ш敝
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class __declspec(uuid("{d33db5ae-0d7b-11ec-82a8-0242ac130003}"))SAudio3FilterXapo : public CXAPOParametersBase
{
public:
	SAudio3FilterXapo();
	~SAudio3FilterXapo() {};

	// ���̓t�H�[�}�b�g�Əo�̓t�H�[�}�b�g�̐ݒ�
	STDMETHOD(LockForProcess)
		(UINT32 inputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pInputLockedParameters,
			UINT32 outputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pOutputLockedParameters)override;

	// �p�����[�^�[�`�F�b�N
	STDMETHOD_(void, SetParameters)
		(_In_reads_bytes_(ParameterByteSize) const void* pParameters, UINT32 ParameterByteSize)override;

	// �p�����[�^�[�̍ŏI�`�F�b�N
	virtual void OnSetParameters
	(const void* pParameters, UINT32 ParameterByteSize)override;

	// �G�t�F�N�g����
	STDMETHOD_(void, Process)
		(UINT32 inputProcessParameterCount,
			const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
			UINT32 outputProcessParameterCount,
			XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
			BOOL isEnabled)override;

private:
	// �G�t�F�N�^�[�̏��E����
	static XAPO_REGISTRATION_PROPERTIES registrationProperties;

	// �t�H�[�}�b�g
	WAVEFORMATEX inputFormat;
	WAVEFORMATEX outputFormat;

	// �t�B���^�W��
	float alpha;
	float omega;
	float a[3];
	float b[3];

	// �v�Z�p
	float inputBackup[2];
	float outputBackup[2];
	
	// �p�����[�^
	SAudio3FilterParameter xapoParameter[3];

	// �t�B���^�[����
	float BiQuadFilter(float input);		// BiQuad(�o2��)�t�B���^�[
};