//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "SampleRateNormalizer.h"
#include "cudaCalc.cuh"

//===================================================================================================================================
// �T���v�����O���g���̐��K��
//===================================================================================================================================
short *SampleRateNormalizer::SetSampleRate(SoundResource *soundResource, int sampleRate)
{
	// GPU�v�Z�p�N���X
	CUDA_CALC *cudaCalc = new CUDA_CALC;

	// [CUDA]���K��
	cudaCalc->normalizer(soundResource->data, soundResource->size, soundResource->waveFormatEx.nChannels,
		soundResource->waveFormatEx.nSamplesPerSec, sampleRate, 2.0f);

	// ��Еt��
	SAFE_DELETE(cudaCalc)
	
	return nullptr;
}