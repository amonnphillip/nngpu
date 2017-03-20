#pragma once

class InputLayerConfig
{
public:
	InputLayerConfig(int width, int height, int depth);
	InputLayerConfig(int size);

	int GetWidth();
	int GetHeight();
	int GetDepth();
private:
	int width;
	int height;
	int depth;
};