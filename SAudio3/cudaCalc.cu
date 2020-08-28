//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "cudaCalc.cuh"

//===================================================================================================================================
// プロトタイプ宣言
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock);
__device__ void Compress(float *fData, short *sData, int compressBlock);

//===================================================================================================================================
// [CPU->GPU]圧縮処理
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock)
{
	Compress(fData, sData, compressBlock);
}

//===================================================================================================================================
// [GPU]圧縮処理
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
// カーネル CPU<->GPU
//===================================================================================================================================
void CUDA_CALC::Kernel(short *_data, long _size, Compress_Data *_compressData)
{
	//スレッドの設定
	int gridSizeX = CUDACalcNS::compressSize / CUDACalcNS::blocksizeX;
	if (gridSizeX < CUDACalcNS::gridMaxX)
	{
		// プロセス
		dim3 grid(gridSizeX, 1, 1);
		dim3 block(CUDACalcNS::blocksizeX, 1, 1);

		// 圧縮量
		_compressData->compressBlock = ((_size / sizeof(short)) / CUDACalcNS::compressSize);

		// デバイスメモリ確保(GPU)
		float *fData = nullptr;
		cudaError_t hr = cudaMalloc((void **)&fData, CUDACalcNS::compressSize * sizeof(float));
		hr = cudaMemset(fData, 0, CUDACalcNS::compressSize * sizeof(float));
		short *sData = nullptr;
		hr = cudaMalloc((void **)&sData, _size);
		hr = cudaMemset(sData, 0, _size);

		// ホスト->デバイス
		hr = cudaMemcpy(sData, &_data[0], _size, cudaMemcpyHostToDevice);

		_compressData->startTime = timeGetTime();
		CompressWave <<<grid, block>>> (fData, sData, _compressData->compressBlock);
		_compressData->usedTime = timeGetTime() - _compressData->startTime;

		hr = cudaMemcpy(_compressData->data, &fData[0], CUDACalcNS::compressSize * sizeof(float), cudaMemcpyDeviceToHost);

		// 後片付け
		cudaFree(fData);
		cudaFree(sData);
	}
}

//===================================================================================================================================
// テスト用プロット
//===================================================================================================================================
//void CUDA_CALC::tmpPlot(void)
//{
//	ImVec2 plotextent(ImGui::GetContentRegionAvailWidth(), 100);
//	ImGui::PlotLines("", tmpPlotData, 10240, 0, "", FLT_MAX, FLT_MAX, plotextent);
//	ImGui::Text("CUDA usedTime:%d", usedTime);
//}