#include "inputlayerconfig.h"

InputLayerConfig::InputLayerConfig(int width, int height, int depth) : 
	width(width),
	height(height),
	depth(depth)
{
}

InputLayerConfig::InputLayerConfig(int size) :
	width(size),
	height(1),
	depth(1)
{
}

int InputLayerConfig::GetWidth()
{
	return width;
}

int InputLayerConfig::GetHeight()
{
	return height;
}

int InputLayerConfig::GetDepth()
{
	return depth;
}