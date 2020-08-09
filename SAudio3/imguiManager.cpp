//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
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

#if USE_IMGUI_DOCKING
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;		// �h�b�L���O�̎g�p����
#endif

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
	showMainPanel	= true;
	showPlayerPanel = true;
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
// [ImGui]���T�C�Y
//===================================================================================================================================
void ImGuiManager::ReSize(LONG right, LONG bottom)
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(right, bottom));
}

//===================================================================================================================================
// [ImGui]���C���p�l��(���T�C�Y)
//===================================================================================================================================
void ImGuiManager::ShowPanel(bool reSize, RECT mainPanelSize)
{
	// ���T�C�Y
	if (reSize)
	{
		ReSize(mainPanelSize.right, mainPanelSize.bottom);
	}

	// ���C���p�l��
	MainPanel();
}

//===================================================================================================================================
// [ImGui]���C���p�l��
//===================================================================================================================================
void ImGuiManager::ShowPanel()
{
	// ���C���p�l��
	MainPanel();
}

//===================================================================================================================================
// ���C���p�l��
//===================================================================================================================================
void ImGuiManager::MainPanel()
{
	// ���C���p�l��
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	if (ImGui::Begin("SAudio3", &showMainPanel,
		ImGuiWindowFlags_::ImGuiWindowFlags_MenuBar |
		ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_::ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_::ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
		ImGuiWindowFlags_::ImGuiWindowFlags_NoBringToFrontOnFocus))
	{
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("New"))
				{
				}
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}

		//�t���[�����[�g��\��
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	
		// �h�b�L���O
		ImGuiID dockspaceID = ImGui::GetID("HUB_DockSpace");
		ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None /*|ImGuiDockNodeFlags_NoResize*/);
		//ImGui::SetNextWindowDockID(dockspaceID, ImGuiCond_Always);
	}
	ImGui::End();

	// �Đ��p�l��
	PlayerPanel();
}

//===================================================================================================================================
// �Đ��p�l��
//===================================================================================================================================
void ImGuiManager::PlayerPanel()
{
	// ���C���p�l��
	if (ImGui::Begin("Player Panel"/*, (bool *)true, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar*/))
	{
		//�t���[�����[�g��\��
		ImGui::Text("player");
	}
	ImGui::End();
}