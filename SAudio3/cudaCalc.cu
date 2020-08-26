//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "cudaCalc.cuh"
#include "ImguiManager.h"

// Maximum number of threads per block:            1024
// Max dimension size of a thread block(x, y, z) : (1024, 1024, 64)
// Max dimension size of a grid size(x, y, z) : (2147483647, 65535, 65535)
__global__ void CompressWave(float *fData, short *sData);
__device__ void Compress(float *fData, short *sData);

__device__ void Compress(float *fData, short *sData)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	fData[idx] = (float)sData[idx];
}

__global__ void CompressWave(float *fData, short *sData)
{
	Compress(fData, sData);
}

void CUDA_CALC::Kernel1(short *_data, long _size)
{
	//スレッドの設定
	int blocksizeX = 1024;
	int dataNum = _size / sizeof(short);
	int gridSizeX = dataNum / 1024;
	if (gridSizeX < 2147483647)
	{
		// プロセス
		dim3 grid(gridSizeX, 1, 1);
		dim3 block(blocksizeX, 1, 1);

		// カーネル関数の呼び出し
		int startTime = timeGetTime();

		// デバイスメモリ確保(GPU)
		float *fData = nullptr;
		cudaError_t hr = cudaMalloc((void **)&fData, dataNum * sizeof(float));
		hr = cudaMemset(fData, 0, dataNum * sizeof(float));

		short *sData = nullptr;
		hr = cudaMalloc((void **)&sData, _size);
		hr = cudaMemset(sData, 0, _size);

		float *cData = (float *)malloc(dataNum * sizeof(float));
		memset(cData, 0, dataNum * sizeof(float));

		// ホスト->デバイス
		hr = cudaMemcpy(sData, &_data[0], _size, cudaMemcpyHostToDevice);

		CompressWave << <grid, block >> > (fData, sData);

		hr = cudaMemcpy(cData, &fData[0], dataNum * sizeof(float), cudaMemcpyDeviceToHost);

		int usedTime = timeGetTime() - startTime;

		// 後片付け
		free(cData);
		cudaFree(fData);
		cudaFree(sData);
	}
}

void CUDA_CALC::tmpPlot(void)
{
	ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
	ImGui::PlotLines("", tmpPlotData, imGuiPlotManagerNS::compressSize, 0, "", FLT_MAX, FLT_MAX, plotextent);
}