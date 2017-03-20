#pragma once

class SigmoidLayerConfig
{
public:
	SigmoidLayerConfig(int width, int height, int depth);
	SigmoidLayerConfig(int size);

	int GetWidth();
	int GetHeight();
	int GetDepth();
private:
	int width;
	int height;
	int depth;
};
