#include "FFT.h"

//===================================================================================================================================
// FFT
//===================================================================================================================================
void FFT::FFTProcess(int samplesCnt, const kiss_fft_cpx in[], kiss_fft_cpx out[])
{
	kiss_fft_cfg cfg;

	if ((cfg = kiss_fft_alloc(samplesCnt, 0, NULL, NULL)) != NULL)
	{
		int cnt = samplesCnt;

		kiss_fft(cfg, in, out);
		SAFE_DELETE(cfg);
	}
	else
	{
		exit(-1);
	}
}