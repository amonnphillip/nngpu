#pragma once

enum LayerDataType {
	Forward = 0,
	Backward = 1
};

struct LayerData
{
public:
	LayerDataType type;
	int width;
	int height;
	int depth;
	double* data;
};

struct LayerDataList
{
public:
	int layerDataCount;
	LayerType layerType;
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


