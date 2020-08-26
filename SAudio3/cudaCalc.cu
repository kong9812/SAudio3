//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "cudaCalc.cuh"
#include "ImguiManager.h"

// Maximum number of threads per block:            1024
// Max dimension size of a thread block(x, y, z) : (1024, 1024, 64)
// Max dimension size of a grid size(x, y, z) : (2147483647, 65535, 65535)
__global__ void CompressWave(float *fData, short *sData, int compressBlock);
__device__ void Compress(float *fData, short *sData, int compressBlock);
__global__ void CompressWave(float *fData, short *sData, int compressBlock)
{
	Compress(fData, sData, compressBlock);
}
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


void CUDA_CALC::Kernel1(short *_data, long _size)
{
	//スレッドの設定
	int dataNum = 10240;
	int blocksizeX = 1024;
	int gridSizeX = 10240 / 1024;
	if (gridSizeX < 2147483647)
	{
		// プロセス
		dim3 grid(gridSizeX, 1, 1);
		dim3 block(blocksizeX, 1, 1);

		// 圧縮量
		int compressBlock = ((_size / sizeof(short)) / 10240);

		// デバイスメモリ確保(GPU)
		float *fData = nullptr;
		cudaError_t hr = cudaMalloc((void **)&fData, dataNum * sizeof(float));
		hr = cudaMemset(fData, 0, dataNum * sizeof(float));

		short *sData = nullptr;
		hr = cudaMalloc((void **)&sData, _size);
		hr = cudaMemset(sData, 0, _size);

		// ホスト->デバイス
		hr = cudaMemcpy(sData, &_data[0], _size, cudaMemcpyHostToDevice);

		int startTime = timeGetTime();
		CompressWave <<<grid, block>>> (fData, sData, compressBlock);
		usedTime = timeGetTime() - startTime;

		hr = cudaMemcpy(tmpPlotData, &fData[0], dataNum * sizeof(float), cudaMemcpyDeviceToHost);

		// 後片付け
		cudaFree(fData);
		cudaFree(sData);
	}
}

void CUDA_CALC::tmpPlot(void)
{
	ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
	ImGui::PlotLines("", tmpPlotData, 10240, 0, "", FLT_MAX, FLT_MAX, plotextent);
	ImGui::Text("CUDA usedTime:%d", usedTime);
}