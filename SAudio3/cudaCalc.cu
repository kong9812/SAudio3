//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "cudaCalc.cuh"

//===================================================================================================================================
// プロトタイプ宣言
//===================================================================================================================================
__global__ void ConversionWave(float *fData, short *sData, int allChannel, int processChannel);
__device__ void Conversion(float *fData, short *sData, int allChannel, int processChannel);

__global__ void CompressWave(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);
__device__ void Compress(float *fData, short *sData, int compressBlock, int allChannel, int processChannel);

__global__ void NormalizeWave(short *sData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate, float gain);
__device__ void Normalize(short *sData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate, float gain);

__global__ void FadeWave(short *outData, short *inData, int allChannel, int processChannel, float fadeAddVolume, float fadeMinusVolume,
	int fadeInStartSampling, int fadeInEndSampling, int fadeOutStartSampling, int fadeOutEndSampling);
__device__ void Fade(short *outData, short *inData, int allChannel, int processChannel, float fadeAddVolume, float fadeMinusVolume,
	int fadeInStartSampling, int fadeInEndSampling, int fadeOutStartSampling, int fadeOutEndSampling);

__global__ void CombineWave(short *outData, float *inData, long sampingPerChannel, int newChannel, int oldChannel);
__device__ void Combine(short *outData, float *inData, long sampingPerChannel, int newChannel, int oldChannel);

//===================================================================================================================================
// [CPU->GPU]変換処理
//===================================================================================================================================
__global__ void ConversionWave(float *fData, short *sData, int allChannel, int processChannel)
{
	// [CPU->GPU]変換処理
	Conversion(fData, sData, allChannel, processChannel);
}

//===================================================================================================================================
// [CPU->GPU]変換処理
//===================================================================================================================================
__device__ void Conversion(float *fData, short *sData, int allChannel, int processChannel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	// MAX:1024
	fData[idx] = (float)sData[(idx*allChannel) + processChannel] / (float)SHRT_MAX;
}

//===================================================================================================================================
// カーネル 変換処理
//===================================================================================================================================
Conversion_Data CUDA_CALC::conversion(short *_data, long _size, int channel)
{
	Conversion_Data tmpConversionData = { NULL };
	tmpConversionData.channel = channel;

	// 1チャンネル当たりのサンプリング数
	tmpConversionData.sampingPerChannel = _size / sizeof(short) / channel;
	
	// ブロック(スレッドX,スレッドY,スレッドZ)
	dim3 block(CUDACalcNS::threadX, 1, 1);
	// グリッド(ブロックX,ブロックY)
	dim3 grid(tmpConversionData.sampingPerChannel / block.x, 1, 1);

	// デバイスメモリ確保(GPU)
	float *fData = nullptr;
	cudaError_t hr = cudaMalloc((void **)&fData, sizeof(float)*tmpConversionData.sampingPerChannel);

	short *sData = nullptr;
	hr = cudaMalloc((void **)&sData, _size);
	hr = cudaMemset(sData, 0, _size);

	// ホスト->デバイス
	hr = cudaMemcpy(sData, &_data[0], _size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// カーネル+デバイス->ホスト
	tmpConversionData.startTime = timeGetTime();
	tmpConversionData.data = new float *[tmpConversionData.channel];
	for (int i = 0; i < tmpConversionData.channel; i++)
	{
		// メモリリセット
		hr = cudaMemset(fData, NULL, sizeof(float)*tmpConversionData.sampingPerChannel);

		tmpConversionData.data[i] = new float[tmpConversionData.sampingPerChannel];
		memset(tmpConversionData.data[i], NULL, sizeof(float)*tmpConversionData.sampingPerChannel);
		// カーネル
		ConversionWave << <grid, block >> > (fData, sData, channel, i);
		hr = cudaMemcpy(tmpConversionData.data[i], &fData[0], sizeof(float)*tmpConversionData.sampingPerChannel, cudaMemcpyDeviceToHost);
	}
	tmpConversionData.usedTime = timeGetTime() - tmpConversionData.startTime;

	// 後片付け
	hr = cudaFree(fData);
	hr = cudaFree(sData);

	return tmpConversionData;
}

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

	float tmpData = 0;
	for (int i = 0; i < compressBlock; i++)
	{
		tmpData += (float)sData[((i + (idx * compressBlock))*allChannel) + processChannel] / (float)SHRT_MAX;
	}
	fData[idx] = tmpData / (float)compressBlock;
}

//===================================================================================================================================
// カーネル 圧縮処理
//===================================================================================================================================
Compress_Data CUDA_CALC::compressor(short *_data, long _size, int channel)
{
	Compress_Data tmpCompressData = { NULL };
	tmpCompressData.channel = channel;
	tmpCompressData.max = 0;
	tmpCompressData.min = 0;

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
	for (int i = 0; i < channel; i++)
	{
		for (int j = 0; j < CUDACalcNS::compressSize; j++)
		{
			if (tmpCompressData.data[i][j] > tmpCompressData.max)
			{
				tmpCompressData.max = tmpCompressData.data[i][j];
			}
			if (tmpCompressData.data[i][j] < tmpCompressData.min)
			{
				tmpCompressData.min = tmpCompressData.data[i][j];
			}
		}
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
__global__ void NormalizeWave(short *outData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate, float gain)
{
	Normalize(outData, inData, inSize, allChannel, oldSampleRate, newSampleRate, gain);
}

//===================================================================================================================================
// [GPU]正規化
//===================================================================================================================================
__device__ void Normalize(short *outData, short *inData, long inSize, int allChannel, int oldSampleRate, int newSampleRate, float gain)
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
			
			// オーバー風呂対策(音量上げる・下げる)
			if (gain > 0)
			{
				// 上げる
				if(gain > 1)
				{ 
					if (((tmp3 > 0) && ((tmp3*gain)) < tmp3) || ((tmp3*gain) > SHRT_MAX))
					{
						tmp3 = SHRT_MAX;
					}
					else if ((tmp3 < 0) && ((tmp3*gain) > tmp3) || ((tmp3*gain) < SHRT_MIN))
					{
						tmp3 = SHRT_MIN;
					}
					else
					{
						tmp3 *= gain;
					}
				}
			}

			//short tmp4 = (short)roundf(tmp3);
			//if (tmp4 > SHRT_MAX)
			//{
			//	tmp4 = SHRT_MAX;
			//}
			//else if (tmp4 < SHRT_MIN)
			//{
			//	tmp4 = SHRT_MIN;
			//}

			outData[idx * allChannel + j] = tmp3;
		}
	}
}

//===================================================================================================================================
// 正規化
//===================================================================================================================================
Normalize_Data CUDA_CALC::normalizer(short *_data, long _size, int channel, int oldSampleRate, int newSampleRate, float gain)
{
	Normalize_Data tmpNormalizeData{ NULL };

	// 音の長さ = 波形のサイズ / 一秒あたりの周波数 / チャネル数 / short型のサイズ
	float soundLengh = ((float)_size / oldSampleRate / channel / sizeof(short));

	// 正規化後のサイズ・周波数
	tmpNormalizeData.newSize = (long)(newSampleRate * channel * soundLengh * sizeof(short));
	tmpNormalizeData.newSampleRate = newSampleRate;

	// デバイスメモリ確保(GPU)
	short *outData = nullptr;
	cudaError_t hr = cudaMalloc((void **)&outData, tmpNormalizeData.newSize);
	hr = cudaMemset(outData, 0, tmpNormalizeData.newSize);

	short *inData = nullptr;
	hr = cudaMalloc((void **)&inData, _size);
	hr = cudaMemset(inData, 0, _size);
	hr = cudaMemcpy(inData, &_data[0], _size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// ブロック(スレッドX,スレッドY,スレッドZ)
	dim3 block(CUDACalcNS::threadX, 1, 1);
	// グリッド(ブロックX,ブロックY)
	dim3 grid(tmpNormalizeData.newSize / channel / sizeof(short) / block.x, 1, 1);

	// [CUDA]正規化
	NormalizeWave << <grid, block >> > (outData, inData, tmpNormalizeData.newSize, channel, oldSampleRate, newSampleRate, gain);

	// 仮データ
	tmpNormalizeData.newData = new short[tmpNormalizeData.newSize / sizeof(short)];
	memset(tmpNormalizeData.newData, NULL, tmpNormalizeData.newSize);
	hr = cudaMemcpy(tmpNormalizeData.newData, &outData[0], tmpNormalizeData.newSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	return tmpNormalizeData;
}


//===================================================================================================================================
// [CPU->GPU]フェイド
//===================================================================================================================================
__global__ void FadeWave(short *outData, short *inData, int allChannel, int processChannel, float fadeAddVolume, float fadeMinusVolume,
	int fadeInStartSampling, int fadeInEndSampling, int fadeOutStartSampling, int fadeOutEndSampling)
{
	Fade(outData, inData, allChannel, processChannel,
		fadeAddVolume, fadeMinusVolume,
		fadeInStartSampling, fadeInEndSampling,
		fadeOutStartSampling, fadeOutEndSampling);
}

//===================================================================================================================================
// [GPU]フェイド
//===================================================================================================================================
__device__ void Fade(short *outData, short *inData, int allChannel, int processChannel, float fadeAddVolume, float fadeMinusVolume,
	int fadeInStartSampling, int fadeInEndSampling, int fadeOutStartSampling, int fadeOutEndSampling)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if ((idx >= fadeInStartSampling) &&
		(idx <= fadeInEndSampling))
	{
		// フェイド中の位置
		int fadeIdx = idx - fadeInStartSampling;

		// ボリューム計算
		float volume = (fadeAddVolume)*fadeIdx;
		outData[idx] = inData[(idx*allChannel) + processChannel] * volume;
	}
	else if ((idx >= fadeOutStartSampling) &&
			(idx <= fadeOutEndSampling))
	{
		// フェイド中の位置
		int fadeIdx = idx - fadeOutStartSampling;

		// ボリューム計算
		float volume = 1.0f - ((fadeMinusVolume)*fadeIdx);
		outData[idx] = inData[(idx*allChannel) + processChannel] * volume;
	}
	else
	{
		outData[idx] = inData[(idx*allChannel) + processChannel];
	}
}

//===================================================================================================================================
// フェイド
//===================================================================================================================================
Fade_Data CUDA_CALC::fade(Normalize_Data normalizeData, int channel, SAudio3FadeParameter fadeParameter)
{
	// 修正:channelごとじゃないから　CUDA関数でチャンネル分けをしないといけない

	Fade_Data fadeData = { NULL };
	fadeData.newSize = normalizeData.newSize;

	// フェイドイン
	int fadeInStartSampling = MS_TO_SAMPLING(fadeParameter.fadeInStartMs, normalizeData.newSampleRate*channel);
	int fadeInEndSampling = MS_TO_SAMPLING(fadeParameter.fadeInEndMs, normalizeData.newSampleRate*channel);
	float fadeAddVolume = 1.0f / ((float)fadeInEndSampling - (float)fadeInStartSampling);
	// フェイドアウト
	int fadeOutStartSampling = MS_TO_SAMPLING(fadeParameter.fadeOutStartMs, normalizeData.newSampleRate*channel);
	int fadeOutEndSampling = MS_TO_SAMPLING(fadeParameter.fadeOutEndMs, normalizeData.newSampleRate*channel);
	float fadeMinusVolume = 1.0f / ((float)fadeOutEndSampling - (float)fadeOutStartSampling);

	// 1チャンネル当たりのサンプリング数
	long sampingPerChannel = normalizeData.newSize / sizeof(short) / channel;
	// ブロック(スレッドX,スレッドY,スレッドZ)
	dim3 block(CUDACalcNS::threadX, 1, 1);
	// グリッド(ブロックX,ブロックY)
	dim3 grid(sampingPerChannel / block.x, 1, 1);

	// デバイスメモリ確保(GPU)
	short *outData = nullptr;
	cudaError_t hr = cudaMalloc((void **)&outData, sizeof(short)*sampingPerChannel);
	short *sData = nullptr;
	hr = cudaMalloc((void **)&sData, normalizeData.newSize);
	hr = cudaMemset(sData, 0, normalizeData.newSize);

	// ホスト->デバイス
	hr = cudaMemcpy(sData, &normalizeData.newData[0], normalizeData.newSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

	// カーネル+デバイス->ホスト
	fadeData.startTime = timeGetTime();
	fadeData.newData = new short[normalizeData.newSize / sizeof(short)];
	memset(fadeData.newData, NULL, normalizeData.newSize);

	for (int i = 0; i < channel; i++)
	{	// メモリリセット
		hr = cudaMemset(outData, NULL, sizeof(short)*sampingPerChannel);
		// カーネル
		FadeWave << <grid, block >> > (outData, sData, channel, i,
			fadeAddVolume, fadeMinusVolume,
			fadeInStartSampling, fadeInEndSampling,
			fadeOutStartSampling, fadeOutEndSampling);
		// 後ろに追加
		hr = cudaMemcpy(&fadeData.newData[sampingPerChannel*i], &outData[0], sizeof(short)*sampingPerChannel, cudaMemcpyDeviceToHost);
	}
	fadeData.usedTime = timeGetTime() - fadeData.startTime;

	// 後片付け
	hr = cudaFree(outData);
	hr = cudaFree(sData);

	return fadeData;
}

//===================================================================================================================================
// [CPU->GPU]合成
//===================================================================================================================================
__global__ void CombineWave(short *outData, float *inData, long sampingPerChannel, int newChannel, int oldChannel)
{
	Combine(outData, inData, sampingPerChannel, newChannel, oldChannel);
}

//===================================================================================================================================
// [GPU]合成
//===================================================================================================================================
__device__ void Combine(short *outData, float *inData, long sampingPerChannel,int newChannel, int oldChannel)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < oldChannel; i++)
	{
		outData[idx * newChannel + i] =
			(short)(inData[sampingPerChannel * i + idx] * (float)SHRT_MAX);
	}
}

//===================================================================================================================================
// 合成(チャンネル数増やすこともできる！)
//===================================================================================================================================
short *CUDA_CALC::combine(float **inData, long sampingPerChannel, int oldChannel, int channel)
{
	// ※サウンドのサイズ:サウンドのサイズ*新しいチャンネル数
	long newSize = sampingPerChannel * channel * sizeof(short);
	short *outData = new short[newSize / sizeof(short)];
	memset(outData, 0, newSize);

	// 合成前のサウンド
	float *tmpInData = new float[sampingPerChannel*oldChannel];
	// 前後で接続 ch1,ch2,ch3...
	for (int i = 0; i < oldChannel; i++)
	{
		memcpy(&tmpInData[sampingPerChannel*i], &inData[i][0],
			sizeof(float)*sampingPerChannel);
	}

	// ブロック(スレッドX,スレッドY,スレッドZ)
	dim3 block(CUDACalcNS::threadX, 1, 1);
	// グリッド(ブロックX,ブロックY)
	dim3 grid(sampingPerChannel / block.x, 1, 1);

	// デバイスメモリ確保(GPU)
	short *outDdata = nullptr;
	cudaError_t hr = cudaMalloc((void **)&outDdata, newSize);
	hr = cudaMemset(outDdata, 0, newSize);
	float *inDdata = nullptr;
	hr = cudaMalloc((void **)&inDdata, sizeof(float)*sampingPerChannel*oldChannel);
	hr = cudaMemset(inDdata, 0, sizeof(float)*sampingPerChannel*oldChannel);

	// ホスト->デバイス
	hr = cudaMemcpy(inDdata, &tmpInData[0], sizeof(float)*sampingPerChannel*oldChannel, cudaMemcpyKind::cudaMemcpyHostToDevice);

	CombineWave << <grid, block >> > (outDdata, inDdata, sampingPerChannel, channel, oldChannel);

	// デバイス->ホスト
	hr = cudaMemcpy(outData, &outDdata[0], newSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	// 後片付け
	hr = cudaFree(outDdata);
	hr = cudaFree(inDdata);
	SAFE_DELETE_ARRAY(tmpInData);

	return outData;
}