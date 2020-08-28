#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "SoundBase.h"
#include "cudaCalc.cuh"

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class ImGuiPlotManager
{
public:
	ImGuiPlotManager();
	~ImGuiPlotManager();

	// ���k�g�`�̃v���b�g����
	void PlotCompressWave(std::string soundName, SoundResource *soundResource);
	
	// �f�[�^�J�E���^�[���Z�b�g
	//void ResetDataCnt(void) { dataCnt = imGuiPlotManagerNS::dataLoadInOneFrame; }

private:
	CUDA_CALC *cudaCalc;
	int dataCnt;										// �c�菈���ł���f�[�^��
	std::map<std::string, Compress_Data> compressData;	// [�A�z�z��]���k�g�`�̃f�[�^

	// ���k�f�[�^�̏���
	//bool InitCompressData(std::string soundName, SoundResource *soundResource);

	// �g�`�̈��k����
	//void CreateCompressWave(std::string soundName, SoundResource *soundResource);
};