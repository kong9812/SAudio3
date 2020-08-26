#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "cudaCalc.cuh"

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
namespace imGuiPlotManagerNS
{
	const int dataLoadInOneFrame = 1000;	// 1�t���[���œǂݍ��߂�f�[�^��
	const int compressSize = 10240;			// ���k��̃f�[�^��(�v���b�g�ł���f�[�^��)	long_max > shrt_max*compressSize
}

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct Compress_Data
{
	int startTime;
	int endTime;
	int usedTime;
	int compressBlock;									// dataSize/imGuiPlotManagerNS::compressSize
	long compressingData;								// �������̃f�[�^
	long readPos;										// �����ʒu
	int dataPos;										// ���k�f�[�^�̏����ʒu
	float data[imGuiPlotManagerNS::compressSize];		// ���k�f�[�^
};

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
	void ResetDataCnt(void) { dataCnt = imGuiPlotManagerNS::dataLoadInOneFrame; }

private:
	CUDA_CALC *cudaCalc;
	int dataCnt;										// �c�菈���ł���f�[�^��
	std::map<std::string, Compress_Data> compressData;	// [�A�z�z��]���k�g�`�̃f�[�^

	// ���k�f�[�^�̏���
	bool InitCompressData(std::string soundName, SoundResource *soundResource);

	// �g�`�̈��k����
	void CreateCompressWave(std::string soundName, SoundResource *soundResource);
};