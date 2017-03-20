#pragma once

class OutputLayerConfig
{
public:
	OutputLayerConfig(int width, int height, int depth);
	OutputLayerConfig(int size);

	int GetWidth();
	int GetHeight();
	int GetDepth();
private:
	int width;
	int height;
	int depth;
};