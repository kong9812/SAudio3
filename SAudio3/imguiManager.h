#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "Directx11.h"
#include "TextureBase.h"
#include "SoundBase.h"
#include "XAudio2Manager.h"
#include "ImGuiPlotManager.h"
#include "ImGuiMixerManager.h"

#if SAUDIO3_TEST_VER
#include "SAudio3FilterXapo.h"
#include "FFT.h"
#endif

//===================================================================================================================================
// �r���h�X�C�b�`
//===================================================================================================================================
// �h�b�L���O�̎g�p��(DONT SWITCH TO FALSE!!!!!)
// true�̂�
#define USE_IMGUI_DOCKING (true)
#if USE_IMGUI_DOCKING
#include "imGui/docking/imgui.h"
#include "imGui/docking/imgui_impl_win32.h"
#include "imGui/docking/imgui_impl_dx11.h"
#include "imGui/docking/imgui_internal.h"

// �v���b�g�̊g��
#include "ImGui/docking/implot.h"
#include "imGui/docking/implot_internal.h"
#include "ImGui/docking/imgui_plot.h"

#else
#include "imGui/imgui.h"
#include "imGui/imgui_impl_win32.h"
#include "imGui/imgui_impl_dx11.h"
#endif

//===================================================================================================================================
// �萔��`
//===================================================================================================================================
namespace imGuiManagerNS
{
	const ImVec2 buttonSize = ImVec2(25, 25);
	const float masteringVoiceVolumeSliderWidth = 400;
	const ImVec2 soundBasePanelProgressBarSize = ImVec2(-1.0f, 5.0f);
#if SAUDIO3_TEST_VER
	const int startHz = 20;
	const int FFTBarCnt = 9;
	const int fftHz[FFTBarCnt] = { 50,100,200,500,1000,2000,5000,10000,30000 };
#endif
}

//===================================================================================================================================
// �\����
//===================================================================================================================================
struct Performance_Record
{
	int cpuNum;
	SYSTEM_INFO systeminfo;
	MEMORYSTATUSEX memoryStatusEx;
	PDH_HQUERY pdhHquery;
	PDH_HCOUNTER cpuCounter;
	PDH_HCOUNTER memoryCounter;
	float cpuUsage;
	float memoryUsage;
	int timeCnt;
};

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class ImGuiManager
{
public:
	ImGuiManager(HWND hWnd, ID3D11Device *device,
		ID3D11DeviceContext *deviceContext, TextureBase *textureBase,
		SoundBase *soundBase, XAudio2Manager *xAudio2Manager);
	~ImGuiManager();

	// [ImGui]�V�����t���[���̍쐬
	void CreateNewFrame();

	// [ImGui]�w���v�}�[�N
	void HelpMarker(const char* desc);

	// [ImGui]�p�l���̕\��
	void ShowPanel(bool reSize, RECT mainPanelSize);
	void ShowPanel();

private:
	TextureBase			*textureBase;		// �e�N�X�`���x�[�X
	SoundBase			*soundBase;			// �T�E���h�x�[�X
	XAudio2Manager		*xAudio2Manager;	// XAudio2�}�l�[�W���[
	ImGuiPlotManager	*imGuiPlotManager;	// [ImGui]�v���b�g�}�l�[�W���[
	ImGuiMixerManager	*imGuiMixerManager;	// [ImGui]�~�N�T�[�}�l�[�W���[

	bool showTestPopup;
	bool showImDemo;					// [ImPlot�EImGui]�f�����
	bool showMainPanel;					// [ImGui�t���O]���C���p�l��
	bool showPlayerPanel;				// [ImGui�t���O]�Đ��p�l��
	bool showSoundBasePanel;			// [ImGui�t���O]�T�E���h�x�[�X�p�l��
	bool showMixerPanel;				// [ImGui�t���O]�T�E���h�x�[�X�p�l��
	bool isPlaying;						// [�v���C���[�p�l��]�Đ���??
	bool isMasteringVoiceVolumeOver1;	// �}�X�^�[�{�C�X�̃{�����[����1�𒴂�����?

	Performance_Record performanceRecord;	// �p�t�H�[�}���X�L�^

#if SAUDIO3_TEST_VER
	int lastFramePlayedSamples;			// �O�t���[���`���t���[�� �Đ������T���v����
	long lastFrameSample;				// �O�t���[���ɋL�^�����ʒu
	int writePos;
	kiss_fft_cpx *inL;
	kiss_fft_cpx *out;
	float *FFT_R;
#endif
											// [ImGui]���T�C�Y
	void ReSize(LONG right, LONG bottom);

	// ���C���p�l��
	void MainPanel();

	// �p�t�H�[�}���X�r���[�A�̏�����
	void PerformanceViewerInit();

	// �p�t�H�[�}���X�r���[�A
	void PerformanceViewer();

	// ���j���[�o�[
	void MenuBar();

	// �Đ��p�l��
	void PlayerPanel();

	// �}�X�^�[�{�C�X�{�����[��
	void MasteringVoiceVolumePanel();

	// �T�E���h�x�[�X�p�l��
	void SoundBasePanel();
};