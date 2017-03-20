#include "relulayerconfig.h"

ReluLayerConfig::ReluLayerConfig(int width, int height, int depth) :
	width(width),
	height(height),
	depth(depth)
{
}

int ReluLayerConfig::GetWidth()
{
	return width;
}

int ReluLayerConfig::GetHeight()
{
	return height;
}

int ReluLayerConfig::GetDepth()
{
	return depth;
}