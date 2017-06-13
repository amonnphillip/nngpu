#pragma once

enum LayerDataType {
	Forward = 0,
	Backward = 1,
	ConvForwardFilter = 2,
	ConvBackFilter = 3,
	PoolBackData = 4
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
		delete layerData;
	}
};


