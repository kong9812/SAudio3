#pragma once
//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "Main.h"
#include "Window.h"
#include "Directx11.h"

//===================================================================================================================================
// クラス
//===================================================================================================================================
class TextureBase
{
public:
	TextureBase(ID3D11Device *device);
	~TextureBase();

	// シェーダーリソースの取り出し
	ID3D11ShaderResourceView* GetShaderResource(char *fileName);

protected:
	std::map<std::string, ID3D11ShaderResourceView*> shaderResource;	// [連想配列]シェーダーリソース

	// テクスチャローダー
	bool LoadTexture(ID3D11Device *device, const char *path, ID3D11ShaderResourceView **shaderResource);
};