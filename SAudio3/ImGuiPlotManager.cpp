//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "ImguiManager.h"
#include "ImGuiPlotManager.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
ImGuiPlotManager::ImGuiPlotManager()
{
	// ������
	dataCnt = imGuiPlotManagerNS::dataLoadInOneFrame;	// �c�菈���ł���f�[�^��

	cudaCalc = new CUDA_CALC;
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
ImGuiPlotManager::~ImGuiPlotManager()
{
	// �A�z�z��̍폜
	if (compressData.size() > NULL)
	{
		compressData.clear();
	}

	SAFE_DELETE(cudaCalc)
}

//===================================================================================================================================
// ���k�g�`�̃v���b�g����
//===================================================================================================================================
void ImGuiPlotManager::PlotCompressWave(std::string soundName, SoundResource *soundResource)
{
	if (!soundResource->isWaveUpdate && soundResource->isCompressed)
	{
		// �v���b�g
		Compress_Data *tmpCompressData = &compressData[soundName];

		ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
		ImGui::PlotLines("", tmpCompressData->data, imGuiPlotManagerNS::compressSize, 0, "", FLT_MAX, FLT_MAX, plotextent);
		tmpCompressData->usedTime = tmpCompressData->endTime - tmpCompressData->startTime;
		ImGui::Text("usedTime:%d", tmpCompressData->usedTime);
	}
	else
	{
		// ���k����
		Compress_Data *tmpCompressData = &compressData[soundName];
		static bool ffgagf = true;
		if (ffgagf)
		{
			cudaCalc->Kernel1(soundResource->data, soundResource->size);
			ffgagf = !ffgagf;
		}
		CreateCompressWave(soundName, soundResource);
		ImGui::Text("compressBlock:%d", tmpCompressData->compressBlock);
		ImGui::Text("readPos:%ld", tmpCompressData->readPos);
		ImGui::Text("dataPos:%d", tmpCompressData->dataPos);
		ImGui::Text("size:%ld", soundResource->size);
	}
}

//===================================================================================================================================
// ���k�f�[�^�̏���
//===================================================================================================================================
bool ImGuiPlotManager::InitCompressData(std::string soundName, SoundResource *soundResource)
{
	// �ĕ`��̕K�v������
	if (soundResource->isWaveUpdate)
	{
		// ���k�������Ă��Ȃ�
		if (!soundResource->isCompressed)
		{
			// [����]���k����(����������Ă��Ȃ���΂����ɓ���)
			Compress_Data *tmpCompressData = &compressData[soundName];
			if (tmpCompressData->compressBlock == NULL)
			{
				// �f�[�^�̃T�C�Y�����k�ʂ�菬����
				if ((soundResource->size / sizeof(short)) < imGuiPlotManagerNS::compressSize)
				{
					tmpCompressData->compressBlock = soundResource->size / sizeof(short);
				}
				else
				{
					tmpCompressData->compressBlock = ((soundResource->size / sizeof(short)) / imGuiPlotManagerNS::compressSize);
				}
				// �����ʒu�̏�����
				tmpCompressData->readPos = NULL;
				tmpCompressData->dataPos = NULL;
				tmpCompressData->startTime = timeGetTime();
			}
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
	return true;
}

//===================================================================================================================================
// �g�`�̈��k����
//===================================================================================================================================
void ImGuiPlotManager::CreateCompressWave(std::string soundName, SoundResource *soundResource)
{
	if (dataCnt <= NULL) { return; }	// �܂����z�����������܂��[(���̃t���[���͂���ȏ㏈���ł��܂���I)

	// ���k�f�[�^�̏���(���k�����K�v�Ȃ��f�[�^���΂�)
	if (InitCompressData(soundName, soundResource))
	{
		Compress_Data *tmpCompressData = &compressData[soundName];

		while (dataCnt > NULL)
		{
			// ���k���������Ȃ炱���ɓ���
			if (tmpCompressData->dataPos < imGuiPlotManagerNS::compressSize)
			{
				// ���k�J�n!
				tmpCompressData->compressingData += soundResource->data[tmpCompressData->readPos];
				tmpCompressData->readPos++;

				// �������肽��
				if ((tmpCompressData->readPos % tmpCompressData->compressBlock) == NULL)
				{
					// �f�[�^�Z�b�g
					tmpCompressData->data[tmpCompressData->dataPos] =
						(float)tmpCompressData->compressingData / (float)tmpCompressData->compressBlock;

					// 0�ɖ߂�
					tmpCompressData->compressingData = NULL;
					// �J�E���g
					tmpCompressData->dataPos++;
				}
				// �J�E���g
				dataCnt--;
			}
			else
			{
				soundResource->isWaveUpdate = false;
				soundResource->isCompressed = true;
				tmpCompressData->readPos = NULL;
				tmpCompressData->dataPos = NULL;
				tmpCompressData->compressingData = NULL;
				tmpCompressData->compressBlock = NULL;
				tmpCompressData->endTime = timeGetTime();
				break;
			}
		}
	}
	return;
}