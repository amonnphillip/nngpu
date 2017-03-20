#pragma once

class FullyConnectedLayerConfig
{
public:
	FullyConnectedLayerConfig(int width, int height, int depth);
	FullyConnectedLayerConfig(int size);

	int GetWidth();
	int GetHeight();
	int GetDepth();
private:
	int width;
	int height;
	int depth;
};

