#include "softmaxlayerconfig.h"

SoftmaxLayerConfig::SoftmaxLayerConfig(int size) :
	size(size)
{
}

int SoftmaxLayerConfig::GetSize()
{
	return size;
}