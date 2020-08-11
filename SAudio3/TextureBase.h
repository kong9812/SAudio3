#pragma once
//===================================================================================================================================
// �C���N���[�h
//===================================================================================================================================
#include "Main.h"
#include "Window.h"
#include "Directx11.h"

//===================================================================================================================================
// �N���X
//===================================================================================================================================
class TextureBase
{
public:
	TextureBase(ID3D11Device *device);
	~TextureBase();

	// �V�F�[�_�[���\�[�X�̎��o��
	ID3D11ShaderResourceView* GetShaderResource(char *fileName);

protected:
	std::map<std::string, ID3D11ShaderResourceView*> shaderResource;	// [�A�z�z��]�V�F�[�_�[���\�[�X

	// �e�N�X�`�����[�_�[
	bool LoadTexture(ID3D11Device *device, const char *path, ID3D11ShaderResourceView **shaderResource);
};