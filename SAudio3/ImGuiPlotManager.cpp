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
	//// ������
	//dataCnt = imGuiPlotManagerNS::dataLoadInOneFrame;	// �c�菈���ł���f�[�^��
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
		auto begin = compressData.begin();
		auto end = compressData.end();
		for (auto i = begin; i != end; i++)
		{
			// [�`�����l������]�f�[�^���̍폜
			for (int j = 0; j < i->second.channel; j++)
			{
				SAFE_DELETE_ARRAY(i->second.data[j])
			}
			SAFE_DELETE(i->second.data)
		}
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
		for (int i = 0; i < soundResource->waveFormatEx.nChannels; i++)
		{
			ImGui::PlotLines("", tmpCompressData->data[i], CUDACalcNS::compressSize, 0, "", FLT_MAX, FLT_MAX, plotextent);
		}
		ImGui::Text("usedTime:%d", tmpCompressData->usedTime);
		ImGui::Text("size:%ld", soundResource->size);
	}
	else
	{
		// ���k����
		//Compress_Data *tmpCompressData = &compressData[soundName];
		//CreateCompressWave(soundName, soundResource);
		//ImGui::Text("compressBlock:%d", tmpCompressData->compressBlock);
		//ImGui::Text("readPos:%ld", tmpCompressData->readPos);
		//ImGui::Text("dataPos:%d", tmpCompressData->dataPos);
		//ImGui::Text("size:%ld", soundResource->size);
		compressData[soundName] = cudaCalc->compressor(soundResource->data, soundResource->size, soundResource->waveFormatEx.nChannels);
		soundResource->isWaveUpdate = !soundResource->isWaveUpdate;
		soundResource->isCompressed = !soundResource->isCompressed;
	}
}

////===================================================================================================================================
//// ���k�f�[�^�̏���
////===================================================================================================================================
//bool ImGuiPlotManager::InitCompressData(std::string soundName, SoundResource *soundResource)
//{
//	// �ĕ`��̕K�v������
//	if (soundResource->isWaveUpdate)
//	{
//		// ���k�������Ă��Ȃ�
//		if (!soundResource->isCompressed)
//		{
//			// [����]���k����(����������Ă��Ȃ���΂����ɓ���)
//			Compress_Data *tmpCompressData = &compressData[soundName];
//			if (tmpCompressData->compressBlock == NULL)
//			{
//				// �f�[�^�̃T�C�Y�����k�ʂ�菬����
//				if ((soundResource->size / sizeof(short)) < imGuiPlotManagerNS::compressSize)
//				{
//					tmpCompressData->compressBlock = soundResource->size / sizeof(short);
//				}
//				else
//				{
//					tmpCompressData->compressBlock = ((soundResource->size / sizeof(short)) / imGuiPlotManagerNS::compressSize);
//				}
//				// �����ʒu�̏�����
//				tmpCompressData->readPos = NULL;
//				tmpCompressData->dataPos = NULL;
//				tmpCompressData->startTime = timeGetTime();
//			}
//		}
//		else
//		{
//			return false;
//		}
//	}
//	else
//	{
//		return false;
//	}
//	return true;
//}

////===================================================================================================================================
//// �g�`�̈��k����
////===================================================================================================================================
//void ImGuiPlotManager::CreateCompressWave(std::string soundName, SoundResource *soundResource)
//{
//	if (dataCnt <= NULL) { return; }	// �܂����z�����������܂��[(���̃t���[���͂���ȏ㏈���ł��܂���I)
//
//	// ���k�f�[�^�̏���(���k�����K�v�Ȃ��f�[�^���΂�)
//	if (InitCompressData(soundName, soundResource))
//	{
//		Compress_Data *tmpCompressData = &compressData[soundName];
//
//		while (dataCnt > NULL)
//		{
//			// ���k���������Ȃ炱���ɓ���
//			if (tmpCompressData->dataPos < imGuiPlotManagerNS::compressSize)
//			{
//				// ���k�J�n!
//				tmpCompressData->compressingData += soundResource->data[tmpCompressData->readPos];
//				tmpCompressData->readPos++;
//
//				// �������肽��
//				if ((tmpCompressData->readPos % tmpCompressData->compressBlock) == NULL)
//				{
//					// �f�[�^�Z�b�g
//					tmpCompressData->data[tmpCompressData->dataPos] =
//						(float)tmpCompressData->compressingData / (float)tmpCompressData->compressBlock;
//
//					// 0�ɖ߂�
//					tmpCompressData->compressingData = NULL;
//					// �J�E���g
//					tmpCompressData->dataPos++;
//				}
//				// �J�E���g
//				dataCnt--;
//			}
//			else
//			{
//				soundResource->isWaveUpdate = false;
//				soundResource->isCompressed = true;
//				tmpCompressData->readPos = NULL;
//				tmpCompressData->dataPos = NULL;
//				tmpCompressData->compressingData = NULL;
//				tmpCompressData->compressBlock = NULL;
//				tmpCompressData->endTime = timeGetTime();
//				break;
//			}
//		}
//	}
//	return;
//}