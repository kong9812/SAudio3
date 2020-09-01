//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "cudaCalc.cuh"

//===================================================================================================================================
// プロトタイプ宣言
//===================================================================================================================================
__global__ void CompressWave(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);
__device__ void Compress(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);
__global__ void NormalizeWave(short *sData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate);
__device__ void Normalize(short *sData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate);

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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	// MAX:1024

	long tmpData = 0;
	for (int i = 0; i < compressBlock; i++)
	{
		tmpData += sData[((i + (idx * compressBlock))*allChannel) + processChannel];
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

	// ブロック(スレッドX,スレッドY,スレッドZ)
	dim3 block(CUDACalcNS::threadX, 1, 1);
	// グリッド(ブロックX,ブロックY)
	dim3 grid(CUDACalcNS::compressSize / block.x, 1, 1);

	// 圧縮量
	tmpCompressData.compressBlock = ((_size / tmpCompressData.channel / sizeof(short)) / CUDACalcNS::compressSize);

	// デバイスメモリ確保(GPU)
	float *fData = nullptr;
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
		CompressWave <<<grid, block>>> (fData, sData, tmpCompressData.compressBlock, channel, i);
		hr = cudaMemcpy(tmpCompressData.data[i], &fData[0], sizeof(float)*CUDACalcNS::compressSize, cudaMemcpyDeviceToHost);
	}
	tmpCompressData.usedTime = timeGetTime() - tmpCompressData.startTime;

	// 後片付け
	hr = cudaFree(fData);
	hr = cudaFree(sData);

	return tmpCompressData;
}

//===================================================================================================================================
// [CPU->GPU]正規化
//===================================================================================================================================
__global__ void NormalizeWave(short *outData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate)
{
	Normalize(outData, inData, inSize, allChannel, oldSampleRate, newSampleRate);
}

//===================================================================================================================================
// [GPU]正規化
//===================================================================================================================================
__device__ void Normalize(short *outData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	// MAX:新しいサンプル数/チャンネル数

	int readPos = (int)((float)idx * ((float)oldSampleRate / (float)newSampleRate));
	float tmpSample = (float)idx * ((float)oldSampleRate / (float)newSampleRate);	// 隙間のサンプル
	tmpSample -= (int)tmpSample;

	for (int j = 0; j < allChannel; j++)
	{
		// 最後のデータ
		if ((idx == (inSize / (int)sizeof(short) / allChannel) - 1))
		{
			outData[idx * allChannel + j] =
				(short)(inData[readPos*allChannel + j]);
		}
		else
		{
			// 次のデータが今のデータより大きいなら
			float tmp1 = inData[readPos * allChannel + j];			// 前のデータ
			float tmp2 = inData[(readPos + 1) * allChannel + j];	// 次のデータ
			float tmp3 = ((tmp2 - tmp1)*tmpSample + tmp1);			// 割合を計算する
			outData[idx * allChannel + j] = (short)roundf(tmp3);
		}

		//short tmp0 = inData[readPos*allChannel + j];
		//float tmp1 = ((short)inData[readPos*allChannel + j]
		//	- (float)inData[readPos*allChannel + j]) * tmpSample;
		//short tmp3 = (short)(inData[readPos*allChannel + j] + ((float)(inData[readPos*allChannel + j] - inData[readPos*allChannel + j]) * tmpSample));
		//outData[idx * allChannel + j] =
		//	(short)(inData[readPos*allChannel + j]);
	}
	// CPU
	//for (int i = 0; i < (wav.data.waveSize / (int)sizeof(short) / wav.fmt.fmtChannel); i++)
	//{
	//	int		readPos = (int)((float)i * (oldSample / (float)wav.fmt.fmtSampleRate));
	//	float	tmpPos = (float)i * (oldSample / (float)wav.fmt.fmtSampleRate);
	//	tmpPos -= (int)tmpPos;
	//	for (int j = 0; j < wav.fmt.fmtChannel; j++)
	//	{
	//		// 最後のデータ
	//		if ((i == (wav.data.waveSize / (int)sizeof(short) / wav.fmt.fmtChannel) - 1))
	//		{
	//			newBuf[i * wav.fmt.fmtChannel + j] =
	//				(short)(wav.data.waveData[readPos*wav.fmt.fmtChannel + j]);
	//		}
	//		else
	//		{
	//			// 次のデータが今のデータより大きいなら
	//			float tmp1 = wav.data.waveData[readPos * wav.fmt.fmtChannel + j];		// 前のデータ
	//			float tmp2 = wav.data.waveData[(readPos + 1) * wav.fmt.fmtChannel + j];	// 次のデータ
	//			float tmp3 = ((tmp2 - tmp1)*tmpPos + tmp1); // 割合を計算する
	//			newBuf[i * wav.fmt.fmtChannel + j] = (short)roundf(tmp3);
	//		}
	//	}
	//}
}

//===================================================================================================================================
// 正規化
//===================================================================================================================================
short *CUDA_CALC::normalizer(short *_data, long _size, int channel, int oldSampleRate, int newSampleRate)
{
	// 音の長さ = 波形のサイズ / 一秒あたりの周波数 / チャネル数 / short型のサイズ
	float soundLengh = ((float)_size / oldSampleRate / channel / sizeof(short));

	// 正規化後のサイズ
	long newSize = (long)(newSampleRate * channel * soundLengh * sizeof(short));

	// デバイスメモリ確保(GPU)
	short *outData = nullptr;
	cudaError_t hr = cudaMalloc((void **)&outData, newSize);
	hr = cudaMemset(outData, 0, newSize);

	short *inData = nullptr;
	hr = cudaMalloc((void **)&inData, _size);
	hr = cudaMemset(inData, 0, _size);
	hr = cudaMemcpy(inData, &_data[0], _size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// ブロック(スレッドX,スレッドY,スレッドZ)
	dim3 block(CUDACalcNS::threadX, 1, 1);
	// グリッド(ブロックX,ブロックY)
	dim3 grid(newSize / channel / sizeof(short) / block.x, 1, 1);

	// [CUDA]正規化
	NormalizeWave << <grid, block >> > (outData, inData, newSize, channel, oldSampleRate, newSampleRate);

	// 仮データ
	short *tmp = new short[newSize / sizeof(short)];
	memset(tmp, NULL, newSize);
	hr = cudaMemcpy(tmp, &outData[0], newSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	SAFE_DELETE(tmp)

	return nullptr;
}