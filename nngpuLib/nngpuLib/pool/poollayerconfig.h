#pragma once

class PoolLayerConfig
{
public:
	PoolLayerConfig(int spatialExtent, int stride);

	int GetSpatialExtent();
	int GetStride();

private:
	int width;
	int height;
	int depth;
	int spatialExtent;
	int stride;
};

