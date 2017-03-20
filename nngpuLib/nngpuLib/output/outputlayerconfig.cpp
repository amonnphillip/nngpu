#include "outputlayerconfig.h"

OutputLayerConfig::OutputLayerConfig(int width, int height, int depth) :
	width(width),
	height(height),
	depth(depth)
{
}

OutputLayerConfig::OutputLayerConfig(int size) :
	width(size),
	height(1),
	depth(1)
{
}

int OutputLayerConfig::GetWidth()
{
	return width;
}

int OutputLayerConfig::GetHeight()
{
	return height;
}

int OutputLayerConfig::GetDepth()
{
	return depth;
}