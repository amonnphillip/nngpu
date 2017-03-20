#include "fullyconnectedlayerconfig.h"

FullyConnectedLayerConfig::FullyConnectedLayerConfig(int width, int height, int depth) :
	width(width),
	height(height),
	depth(depth)
{
}

FullyConnectedLayerConfig::FullyConnectedLayerConfig(int size) :
	width(size),
	height(1),
	depth(1)
{
}

int FullyConnectedLayerConfig::GetWidth()
{
	return width;
}

int FullyConnectedLayerConfig::GetHeight()
{
	return height;
}

int FullyConnectedLayerConfig::GetDepth()
{
	return depth;
}