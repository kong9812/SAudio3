//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "cudaCalc.cuh"

//===================================================================================================================================
// �v���g�^�C�v�錾
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock);
__device__ void Compress(float *fData, short *sData, int compressBlock);

//===================================================================================================================================
// [CPU->GPU]���k����
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock)
{
	Compress(fData, sData, compressBlock);
}

//===================================================================================================================================
// [GPU]���k����
//===================================================================================================================================
__device__ void Compress(float *fData, short *sData, int compressBlock)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long tmpData = 0;
	for (int i = 0; i < compressBlock; i++)
	{
		tmpData += sData[i + (idx * compressBlock)];
	}
	fData[idx] = (float)tmpData / compressBlock;
}

//===================================================================================================================================
// �J�[�l�� CPU<->GPU
//===================================================================================================================================
void CUDA_CALC::Kernel(short *_data, long _size, Compress_Data *_compressData)
{
	//�X���b�h�̐ݒ�
	int gridSizeX = CUDACalcNS::compressSize / CUDACalcNS::blocksizeX;
	if (gridSizeX < CUDACalcNS::gridMaxX)
	{
		// �v���Z�X
		dim3 grid(gridSizeX, 1, 1);
		dim3 block(CUDACalcNS::blocksizeX, 1, 1);

		// ���k��
		_compressData->compressBlock = ((_size / sizeof(short)) / CUDACalcNS::compressSize);

		// �f�o�C�X�������m��(GPU)
		float *fData = nullptr;
		cudaError_t hr = cudaMalloc((void **)&fData, CUDACalcNS::compressSize * sizeof(float));
		hr = cudaMemset(fData, 0, CUDACalcNS::compressSize * sizeof(float));
		short *sData = nullptr;
		hr = cudaMalloc((void **)&sData, _size);
		hr = cudaMemset(sData, 0, _size);

		// �z�X�g->�f�o�C�X
		hr = cudaMemcpy(sData, &_data[0], _size, cudaMemcpyHostToDevice);

		_compressData->startTime = timeGetTime();
		CompressWave <<<grid, block>>> (fData, sData, _compressData->compressBlock);
		_compressData->usedTime = timeGetTime() - _compressData->startTime;

		hr = cudaMemcpy(_compressData->data, &fData[0], CUDACalcNS::compressSize * sizeof(float), cudaMemcpyDeviceToHost);

		// ��Еt��
		cudaFree(fData);
		cudaFree(sData);
	}
}

//===================================================================================================================================
// �e�X�g�p�v���b�g
//===================================================================================================================================
//void CUDA_CALC::tmpPlot(void)
//{
//	ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
//	ImGui::PlotLines("", tmpPlotData, 10240, 0, "", FLT_MAX, FLT_MAX, plotextent);
//	ImGui::Text("CUDA usedTime:%d", usedTime);
//}