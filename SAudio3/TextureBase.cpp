//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "TextureBase.h"
#include "stb_image.h"

//===================================================================================================================================
// �R���X�g���N�^
//===================================================================================================================================
TextureBase::TextureBase(ID3D11Device *device)
{
	WIN32_FIND_DATA FindFileData;	// �t�@�C���f�[�^
	int fileNum = 0;				// �t�@�C����

	// �t�@�C�������n���h��
	HANDLE hFile = FindFirstFileEx("Texture\\*.png*", FindExInfoBasic, &FindFileData,
		FindExSearchNameMatch, NULL, NULL);

	// �t�@�C������
	if (hFile != INVALID_HANDLE_VALUE)
	{
		// �t�@�C���p�X
		std::string path= "Texture\\";
		path += FindFileData.cFileName;

		// �V�F�[�_�[���\�[�X�̍쐬
		if (!LoadTexture(device, path.c_str(), &shaderResource[FindFileData.cFileName]))
		{
			// �G���[���b�Z�[�W
			MessageBox(NULL, errorNS::DXShaderResourceError, APP_NAME, (MB_OK | MB_ICONERROR));

			// �����I��
			PostQuitMessage(0);
		}
		
		// �N���A
		path.empty();
		
		// �t�@�C���J�E���^�[
		fileNum++;

		while (FindNextFile(hFile, &FindFileData))
		{
			// �t�@�C���p�X
			path = "Texture\\";
			path += FindFileData.cFileName;

			// �V�F�[�_�[���\�[�X�̍쐬
			if (!LoadTexture(device, path.c_str(), &shaderResource[FindFileData.cFileName]))
			{
				// �G���[���b�Z�[�W
				MessageBox(NULL, errorNS::DXShaderResourceError, APP_NAME, (MB_OK | MB_ICONERROR));

				// �����I��
				PostQuitMessage(0);
			}

			// �N���A
			path.empty();
			
			// �t�@�C���J�E���^�[
			fileNum++;
		} 

		// ��Еt��
		FindClose(hFile);
		path.clear();
	}
	else
	{
		// �G���[���b�Z�[�W
		MessageBox(NULL, errorNS::TextureImportError, APP_NAME, (MB_OK | MB_ICONERROR));

#ifndef _DEBUG
		// �����I��
		PostQuitMessage(0);
#endif
	}
}

//===================================================================================================================================
// �f�X�g���N�^
//===================================================================================================================================
TextureBase::~TextureBase()
{
	// �A�z�z��̍폜
	if (shaderResource.size() > NULL)
	{
		// �V�F�[�_�[���\�[�X�̏I������
		auto begin = shaderResource.begin();
		auto end = shaderResource.end();
		for (auto i = begin; i != end; i++)
		{
			SAFE_RELEASE(i->second)
		}
		shaderResource.clear();
	}
}

//===================================================================================================================================
// �e�N�X�`�����[�_�[
//===================================================================================================================================
bool TextureBase::LoadTexture(ID3D11Device *device, const char *path , ID3D11ShaderResourceView **shaderResource)
{
	int imageWidth = NULL;	// �摜�̉���
	int imageHeight = NULL;	// �摜�̏c�� 
	unsigned char* imageData = stbi_load(path, &imageWidth, &imageHeight, NULL, 4);	// [STB]�摜�f�[�^�̎��o��
	if (imageData == NULL)
	{
		return false;
	}

	// �e�N�X�`�����
	D3D11_TEXTURE2D_DESC textureDesc;
	ZeroMemory(&textureDesc, sizeof(textureDesc));
	textureDesc.Width = imageWidth;
	textureDesc.Height = imageHeight;
	textureDesc.MipLevels = 1;
	textureDesc.ArraySize = 1;
	textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	textureDesc.SampleDesc.Count = 1;
	textureDesc.Usage = D3D11_USAGE_DEFAULT;
	textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	textureDesc.CPUAccessFlags = 0;

	// �e�N�X�`���f�[�^
	ID3D11Texture2D *tmpTexture = NULL;
	// �T�u���\�[�X�f�[�^
	D3D11_SUBRESOURCE_DATA subResource;
	subResource.pSysMem = imageData;
	subResource.SysMemPitch = textureDesc.Width * 4;
	subResource.SysMemSlicePitch = 0;
	device->CreateTexture2D(&textureDesc, &subResource, &tmpTexture);

	// �V�F�[�_�[���\�[�X���
	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
	ZeroMemory(&srvDesc, sizeof(srvDesc));
	srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = textureDesc.MipLevels;
	srvDesc.Texture2D.MostDetailedMip = 0;
	device->CreateShaderResourceView(tmpTexture, &srvDesc, &*shaderResource);
	SAFE_RELEASE(tmpTexture)

	return true;
}

//===================================================================================================================================
// �V�F�[�_�[���\�[�X�̎��o��
//===================================================================================================================================
ID3D11ShaderResourceView* TextureBase::GetShaderResource(char *fileName)
{
	return shaderResource[fileName];
}