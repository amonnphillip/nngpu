#include "convlayerconfig.h"

ConvLayerConfig::ConvLayerConfig(int filterWidth, int filterHeight, int filterCount, int pad, int stride) :
	filterWidth(filterWidth),
	filterHeight(filterHeight),
	filterCount(filterCount),
	pad(pad),
	stride(stride)
{
}

int ConvLayerConfig::GetFilterWidth()
{
	return filterWidth;
}

int ConvLayerConfig::GetFilterHeight()
{
	return filterHeight;
}

int ConvLayerConfig::GetStride()
{
	return stride;
}

int ConvLayerConfig::GetPad()
{
	return pad;
}

int ConvLayerConfig::GetFilterCount()
{
	return filterCount;
}