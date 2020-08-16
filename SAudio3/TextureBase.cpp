//===================================================================================================================================
// インクルード
//===================================================================================================================================
#include "TextureBase.h"
#include "stb_image.h"

//===================================================================================================================================
// コンストラクタ
//===================================================================================================================================
TextureBase::TextureBase(ID3D11Device *device)
{
	WIN32_FIND_DATA FindFileData;	// ファイルデータ
	int fileNum = 0;				// ファイル数

	// ファイル検索ハンドル
	HANDLE hFile = FindFirstFileEx("Texture\\*.png*", FindExInfoBasic, &FindFileData,
		FindExSearchNameMatch, NULL, NULL);

	// ファイル検索
	if (hFile != INVALID_HANDLE_VALUE)
	{
		// ファイルパス
		std::string path= "Texture\\";
		path += FindFileData.cFileName;

		// シェーダーリソースの作成
		if (!LoadTexture(device, path.c_str(), &shaderResource[FindFileData.cFileName]))
		{
			// エラーメッセージ
			MessageBox(NULL, errorNS::DXShaderResourceError, APP_NAME, (MB_OK | MB_ICONERROR));

			// 強制終了
			PostQuitMessage(0);
		}
		
		// クリア
		path.empty();
		
		// ファイルカウンター
		fileNum++;

		while (FindNextFile(hFile, &FindFileData))
		{
			// ファイルパス
			path = "Texture\\";
			path += FindFileData.cFileName;

			// シェーダーリソースの作成
			if (!LoadTexture(device, path.c_str(), &shaderResource[FindFileData.cFileName]))
			{
				// エラーメッセージ
				MessageBox(NULL, errorNS::DXShaderResourceError, APP_NAME, (MB_OK | MB_ICONERROR));

				// 強制終了
				PostQuitMessage(0);
			}

			// クリア
			path.empty();
			
			// ファイルカウンター
			fileNum++;
		} 

		// 後片付け
		FindClose(hFile);
		path.clear();
	}
	else
	{
		// エラーメッセージ
		MessageBox(NULL, errorNS::TextureImportError, APP_NAME, (MB_OK | MB_ICONERROR));

#ifndef _DEBUG
		// 強制終了
		PostQuitMessage(0);
#endif
	}
}

//===================================================================================================================================
// デストラクタ
//===================================================================================================================================
TextureBase::~TextureBase()
{
	// 連想配列の削除
	if (shaderResource.size() > NULL)
	{
		// シェーダーリソースの終了処理
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
// テクスチャローダー
//===================================================================================================================================
bool TextureBase::LoadTexture(ID3D11Device *device, const char *path , ID3D11ShaderResourceView **shaderResource)
{
	int imageWidth = NULL;	// 画像の横幅
	int imageHeight = NULL;	// 画像の縦幅 
	unsigned char* imageData = stbi_load(path, &imageWidth, &imageHeight, NULL, 4);	// [STB]画像データの取り出し
	if (imageData == NULL)
	{
		return false;
	}

	// テクスチャ情報
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

	// テクスチャデータ
	ID3D11Texture2D *tmpTexture = NULL;
	// サブリソースデータ
	D3D11_SUBRESOURCE_DATA subResource;
	subResource.pSysMem = imageData;
	subResource.SysMemPitch = textureDesc.Width * 4;
	subResource.SysMemSlicePitch = 0;
	device->CreateTexture2D(&textureDesc, &subResource, &tmpTexture);

	// シェーダーリソース情報
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
// シェーダーリソースの取り出し
//===================================================================================================================================
ID3D11ShaderResourceView* TextureBase::GetShaderResource(char *fileName)
{
	return shaderResource[fileName];
}