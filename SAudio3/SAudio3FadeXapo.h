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
	float fadeTime;
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class __declspec(uuid("{3C667D8D-4BA9-487F-8944-57419AB69909}"))SAudio3FadeXapo : public CXAPOParametersBase
{
public:
	SAudio3FadeXapo();
	~SAudio3FadeXapo() {};

	STDMETHOD(LockForProcess)
		(UINT32 inputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pInputLockedParameters,
			UINT32 outputLockedParameterCount,
			const XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS * pOutputLockedParameters)
		override;


	STDMETHOD_(void, Process)
		(UINT32 inputProcessParameterCount,
			const XAPO_PROCESS_BUFFER_PARAMETERS * pInputProcessParameters,
			UINT32 outputProcessParameterCount,
			XAPO_PROCESS_BUFFER_PARAMETERS * pOutputProcessParameters,
			BOOL isEnabled)
		override;

	virtual void OnSetParameters
	(const void* pParameters, UINT32 ParameterByteSize);

private:
	// �G�t�F�N�^�[�̏��E����
	XAPO_REGISTRATION_PROPERTIES registrationProperties;

	// �p�����[�^
	SAudio3FadeParameter xapoParameter[3];
};