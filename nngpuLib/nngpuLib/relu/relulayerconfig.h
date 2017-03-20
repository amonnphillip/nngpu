#pragma once

class ReluLayerConfig
{
public:
	ReluLayerConfig(int width, int height, int depth);

	int GetWidth();
	int GetHeight();
	int GetDepth();
private:
	int width;
	int height;
	int depth;
};

