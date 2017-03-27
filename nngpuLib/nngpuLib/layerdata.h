#pragma once

struct LayerData
{
public:
	LayerType type;
	int width;
	int height;
	int depth;
	double* data;
};

struct LayerDataList
{
public:
	int layerDataCount;
	LayerData* layerData;
	void CleanUp()
	{
		LayerData* l = layerData;
		for (int index = 0; index < layerDataCount; index++)
		{
			delete l;
			l++;
		}
	}
};

enum LayerDataType {
	Forward = 0,
	Backward = 1
};
