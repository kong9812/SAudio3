//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "cudaCalc.cuh"

//===================================================================================================================================
// プロトタイプ宣言
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);
__device__ void Compress(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);

//===================================================================================================================================
// [CPU->GPU]圧縮処理
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock, int allChannel, int processChannel)
{
	// [GPU]圧縮処理
	Compress(fData, sData, compressBlock, allChannel, processChannel);
}

//===================================================================================================================================
// [GPU]圧縮処理
//===================================================================================================================================
__device__ void Compress(float *fData, short *sData, int compressBlock, int allChannel, int processChannel)
{
	// 処理中のチャンネル : blockIdx.x
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	// MAX:1024

	long tmpData = 0;
	for (int i = 0; i < compressBlock; i++)
	{
		tmpData += sData[((i + (threadIdx.x * compressBlock))*allChannel) + processChannel];
	}
	fData[idx] = (float)tmpData / compressBlock;
}

//===================================================================================================================================
// カーネル CPU<->GPU
//===================================================================================================================================
Compress_Data CUDA_CALC::compressor(short *_data, long _size, int channel)
{
	Compress_Data tmpCompressData = { NULL };
	tmpCompressData.channel = channel;

	// プロセス
	dim3 block(CUDACalcNS::threadX, 1, 1);
	dim3 grid(1, 1, 1);

	// 圧縮量
	tmpCompressData.compressBlock = ((_size / tmpCompressData.channel / sizeof(short)) / CUDACalcNS::compressSize);

	// デバイスメモリ確保(GPU)
	float *fData = nullptr;
	size_t pitch = NULL;
	cudaError_t hr = cudaMalloc((void **)&fData, sizeof(float)*CUDACalcNS::compressSize);

	short *sData = nullptr;
	hr = cudaMalloc((void **)&sData, _size);
	hr = cudaMemset(sData, 0, _size);

	// ホスト->デバイス
	hr = cudaMemcpy(sData, &_data[0], _size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// カーネル+デバイス->ホスト
	tmpCompressData.startTime = timeGetTime();
	tmpCompressData.data = new float *[tmpCompressData.channel];
	for (int i = 0; i < tmpCompressData.channel; i++)
	{
		// メモリリセット
		hr = cudaMemset(fData, NULL, sizeof(float)*CUDACalcNS::compressSize);

		tmpCompressData.data[i] = new float[CUDACalcNS::compressSize];
		memset(tmpCompressData.data[i], NULL, sizeof(float)*CUDACalcNS::compressSize);
		// カーネル
		CompressWave << <grid, block >> > (fData, sData, tmpCompressData.compressBlock, channel, i);
		hr = cudaMemcpy(tmpCompressData.data[i], &fData[0], sizeof(float)*CUDACalcNS::compressSize, cudaMemcpyDeviceToHost);
	}
	tmpCompressData.usedTime = timeGetTime() - tmpCompressData.startTime;

	// 後片付け
	hr = cudaFree(fData);
	hr = cudaFree(sData);

	return tmpCompressData;
}

//===================================================================================================================================
// 正規化
//===================================================================================================================================
short *CUDA_CALC::normalizer(short *_data, short sampleRate)
{
	return nullptr;
}