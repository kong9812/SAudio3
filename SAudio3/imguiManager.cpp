//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "imguiManager.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
ImGuiManager::ImGuiManager(HWND hWnd, ID3D11Device *device,
	ID3D11DeviceContext	*deviceContext, TextureBase *_textureBase,
	SoundBase *_soundBase)
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
	ImGui::StyleColorsClassic();

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
	showMainPanel = true;
	showPlayerPanel = true;
	showSoundBasePanel = true;
	isPlaying = false;

	// �e�N�X�`���x�[�X
	textureBase = _textureBase;
	soundBase = _soundBase;
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
	if (showMainPanel)
	{
		ImGui::SetNextWindowBgAlpha(0.7f);
		ImGui::SetNextWindowPos(ImVec2(0, 0));
		ImGui::Begin("SAudio3", &showMainPanel,
			ImGuiWindowFlags_::ImGuiWindowFlags_MenuBar |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_::ImGuiWindowFlags_NoBringToFrontOnFocus);

		// ���j���[�o�[
		MenuBar();

		// �e�X�g�����̕\��
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		// �h�b�L���O
		ImGuiID dockspaceID = ImGui::GetID("MainPanelDockSpace");
		ImGui::DockSpace(dockspaceID, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);
		ImGui::End();

	}

	// �Đ��p�l��
	PlayerPanel();

	// �T�E���h�x�[�X�p�l��
	SoundBasePanel();
}

//===================================================================================================================================
// ���j���[�o�[
//===================================================================================================================================
void ImGuiManager::MenuBar()
{
	if (ImGui::BeginMenuBar())
	{
		// �t�@�C������
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("New")){}
			ImGui::EndMenu();
		}

		// �E�C���h
		if (ImGui::BeginMenu("Window"))
		{
			ImGui::MenuItem("Player", "", &showPlayerPanel);
			ImGui::MenuItem("Sound Base", "", &showSoundBasePanel);		
			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}
}

//===================================================================================================================================
// �Đ��p�l��
//===================================================================================================================================
void ImGuiManager::PlayerPanel()
{
	// �Đ��p�l��
	if (showPlayerPanel)
	{
		ImGui::Begin("Player Panel", &showPlayerPanel, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);

		// �e�X�g�����̕\��
		ImGui::Text("player");

		if (!isPlaying)
		{	// �Đ��{�^��
			if (ImGui::ImageButton((void*)textureBase->GetShaderResource((char *)"playButton.png"), imGuiManagerNS::buttonSize))
			{
				// �Đ�

				isPlaying = true;
			}
		}
		else
		{
			// �ꎞ��~�{�^��
			if (ImGui::ImageButton((void*)textureBase->GetShaderResource((char *)"pauseButton.png"), imGuiManagerNS::buttonSize))
			{
				// �ꎞ��~

				isPlaying = false;
			}
		}
		ImGui::End();
	}
}

//===================================================================================================================================
// �T�E���h�x�[�X�p�l��
//===================================================================================================================================
void ImGuiManager::SoundBasePanel()
{
	// �Đ��p�l��
	if (showSoundBasePanel)
	{
		ImGui::Begin("Sound Base Panel", &showSoundBasePanel, ImGuiWindowFlags_::ImGuiWindowFlags_NoTitleBar);

		// �T�E���h���̕\��&�{�^��
		auto begin = soundBase->soundResource.begin();
		auto end = soundBase->soundResource.end();
		for (auto i = begin; i != end; i++)
		{
			ImGui::Button(i->first.data());
		}

		ImGui::End();
	}
}