#include "imguiManager.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
ImGuiManager::ImGuiManager(HWND hWnd, ID3D11Device *device, ID3D11DeviceContext	*deviceContext)
{
	// �o�[�W�����`�F�b�N
	IMGUI_CHECKVERSION();

	// [ImGui]�R���e�N�X�g�̍쐬
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// �_�[�N���[�h
	ImGui::StyleColorsDark();

	// [ImGui]win32�̏�����
	if (!ImGui_ImplWin32_Init(hWnd))
	{
		// �G���[���b�Z�[�W
		MessageBox(NULL, errorNS::ImGuiWin32InitError, APP_NAME, (MB_OK | MB_ICONERROR));

		// �����I��
		PostQuitMessage(0);
		return;
	}

	// [ImGui]������
	if (!ImGui_ImplDX11_Init(device, deviceContext))
	{
		// �G���[���b�Z�[�W
		MessageBox(NULL, errorNS::ImGuiInitError, APP_NAME, (MB_OK | MB_ICONERROR));

		// �����I��
		PostQuitMessage(0);
		return;
	}

	// ImGui�t���O�̏�����
	showSample = true;
}

//===================================================================================================================================
// [ImGui]�V�����t���[���̍쐬
//===================================================================================================================================
void ImGuiManager::CreateNewFrame()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
ImGuiManager::~ImGuiManager()
{
	// [ImGui]�I������
	ImGui_ImplDX11_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

//===================================================================================================================================
// [ImGui]�e�X�g
//===================================================================================================================================
void ImGuiManager::test()
{
	// �T���v��UI
	if (showSample)
		ImGui::ShowDemoWindow(&showSample);

	
}