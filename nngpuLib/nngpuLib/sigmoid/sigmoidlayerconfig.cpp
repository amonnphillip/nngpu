#include "sigmoidlayerconfig.h"

SigmoidLayerConfig::SigmoidLayerConfig(int width, int height, int depth) :
	width(width),
	height(height),
	depth(depth)
{
}

SigmoidLayerConfig::SigmoidLayerConfig(int size) :
	width(size),
	height(1),
	depth(1)
{
}

int SigmoidLayerConfig::GetWidth()
{
	return width;
}

int SigmoidLayerConfig::GetHeight()
{
	return height;
}

int SigmoidLayerConfig::GetDepth()
{
	return depth;
}