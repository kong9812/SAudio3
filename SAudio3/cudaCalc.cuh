#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#pragma comment(lib, "cudart_static.lib")

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
namespace CUDACalcNS
{
	// GTX 970
	// Maximum number of threads per block:            1024
	// Max dimension size of a thread block(x, y, z) : (1024, 1024, 64)
	// Max dimension size of a grid size(x, y, z) : (2147483647, 65535, 65535)
	const int gridMaxX		= INT_MAX;
	const int threadX		= 1024;
	const int compressSize	= 1024;			// ���k��̃f�[�^��(�v���b�g�ł���f�[�^��)	long_max > shrt_max*compressSize
}

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct Conversion_Data
{
	int sampingPerChannel;
	int startTime;
	int usedTime;
	int channel;
	float **data;
};

struct Compress_Data
{
	int startTime;
	int usedTime;
	int compressBlock;
	int channel;
	float max;
	float min;
	float **data;				// ���k�f�[�^
};

//===================================================================================================================================
// �v���g�^�C�v�錾
//===================================================================================================================================
class CUDA_CALC
{
public:
	CUDA_CALC() {};
	~CUDA_CALC() {};

	// �ϊ�
	Conversion_Data conversion(short *_data, long _size, int channel);

	// ���k
	Compress_Data compressor(short *_data, long _size, int channel);

	// ���K��
	short *normalizer(short *_data, long _size, int channel, int oldSampleRate, int newSampleRate);
};