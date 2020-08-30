//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "cudaCalc.cuh"

//===================================================================================================================================
// �v���g�^�C�v�錾
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);
__device__ void Compress(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);

//===================================================================================================================================
// [CPU->GPU]���k����
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock, int allChannel, int processChannel)
{
	// [GPU]���k����
	Compress(fData, sData, compressBlock, allChannel, processChannel);
}

//===================================================================================================================================
// [GPU]���k����
//===================================================================================================================================
__device__ void Compress(float *fData, short *sData, int compressBlock, int allChannel, int processChannel)
{
	// �������̃`�����l�� : blockIdx.x
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	// MAX:1024

	long tmpData = 0;
	for (int i = 0; i < compressBlock; i++)
	{
		tmpData += sData[((i + (threadIdx.x * compressBlock))*allChannel) + processChannel];
	}
	fData[idx] = (float)tmpData / compressBlock;
}

//===================================================================================================================================
// �J�[�l�� CPU<->GPU
//===================================================================================================================================
Compress_Data CUDA_CALC::compressor(short *_data, long _size, int channel)
{
	Compress_Data tmpCompressData = { NULL };
	tmpCompressData.channel = channel;

	// �v���Z�X
	dim3 block(CUDACalcNS::threadX, 1, 1);
	dim3 grid(1, 1, 1);

	// ���k��
	tmpCompressData.compressBlock = ((_size / tmpCompressData.channel / sizeof(short)) / CUDACalcNS::compressSize);

	// �f�o�C�X�������m��(GPU)
	float *fData = nullptr;
	size_t pitch = NULL;
	cudaError_t hr = cudaMalloc((void **)&fData, sizeof(float)*CUDACalcNS::compressSize);

	short *sData = nullptr;
	hr = cudaMalloc((void **)&sData, _size);
	hr = cudaMemset(sData, 0, _size);

	// �z�X�g->�f�o�C�X
	hr = cudaMemcpy(sData, &_data[0], _size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// �J�[�l��+�f�o�C�X->�z�X�g
	tmpCompressData.startTime = timeGetTime();
	tmpCompressData.data = new float *[tmpCompressData.channel];
	for (int i = 0; i < tmpCompressData.channel; i++)
	{
		// ���������Z�b�g
		hr = cudaMemset(fData, NULL, sizeof(float)*CUDACalcNS::compressSize);

		tmpCompressData.data[i] = new float[CUDACalcNS::compressSize];
		memset(tmpCompressData.data[i], NULL, sizeof(float)*CUDACalcNS::compressSize);
		// �J�[�l��
		CompressWave << <grid, block >> > (fData, sData, tmpCompressData.compressBlock, channel, i);
		hr = cudaMemcpy(tmpCompressData.data[i], &fData[0], sizeof(float)*CUDACalcNS::compressSize, cudaMemcpyDeviceToHost);
	}
	tmpCompressData.usedTime = timeGetTime() - tmpCompressData.startTime;

	// ��Еt��
	hr = cudaFree(fData);
	hr = cudaFree(sData);

	return tmpCompressData;
}

//===================================================================================================================================
// ���K��
//===================================================================================================================================
short *CUDA_CALC::normalizer(short *_data, short sampleRate)
{
	return nullptr;
}