#include "poollayerconfig.h"

PoolLayerConfig::PoolLayerConfig(int spatialExtent, int stride) :
	spatialExtent(spatialExtent),
	stride(stride)
{
}

int PoolLayerConfig::GetSpatialExtent()
{
	return spatialExtent;
}

int PoolLayerConfig::GetStride()
{
	return stride;
}