#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "XAudio2Manager.h"

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct SAudio3FadeParameter
{
	long	allSampling;
	int		fadeInPosMs;	// [�t�F�C�h�C��]�t�F�[�h�̊J�n����(ms)
	int		fadeOutPosMs;	// [�t�F�C�h�A�E�g]�t�F�[�h�̊J�n����(ms)
	int		fadeInMs;		// [�t�F�C�h�C��]�t�F�C�h�̎�������(ms)
	int		fadeOutMs;		// [�t�F�C�h�A�E�g]�t�F�C�h�̎���(ms)
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class __declspec(uuid("{3C667D8D-4BA9-487F-8944-57419AB69909}"))SAudio3FadeXapo : public CXAPOParametersBase
{
public:
	SAudio3FadeXapo();
	~SAudio3FadeXapo() {};

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

	// ���M�p
	STDMETHOD_(void, GetParameters)
		(_Out_writes_bytes_(ParameterByteSize) void* pParameters, UINT32 ParameterByteSize)override;

private:
	// �G�t�F�N�^�[�̏��E����
	static XAPO_REGISTRATION_PROPERTIES registrationProperties;
	
	// �t�H�[�}�b�g
	WAVEFORMATEX inputFormat;
	WAVEFORMATEX outputFormat;

	int		samplingCnt;			// �T���v�����O�J�E���^�[(�����ʒu)
	float	fadeAddVolume;			// [�t�F�C�h�C��]1�T���v�����O������̃{�����[��
	float	fadeMinusVolume;		// [�t�F�C�h�A�E�g]1�T���v�����O������̃{�����[��
	int		fadeInPosSampling;		// [�t�F�C�h�C��]�t�F�[�h�̊J�n�ʒu(�T���v�����O)
	int		fadeOutPosSampling;		// [�t�F�C�h�A�E�g]�t�F�[�h�̊J�n�ʒu(�T���v�����O)
	int		fadeInSampling;			// [�t�F�C�h�C��]�t�F�C�h�̎����T���v�����O��
	int		fadeOutSampling;		// [�t�F�C�h�A�E�g]�t�F�C�h�̎����T���v�����O��

	// �p�����[�^
	SAudio3FadeParameter xapoParameter[3];
};