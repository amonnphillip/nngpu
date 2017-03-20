#pragma once

class ConvLayerConfig
{
public:
	ConvLayerConfig(int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride);

	int GetFilterWidth();
	int GetFilterHeight();
	int GetFilterDepth();
	int GetStride();
	int GetPad();
	int GetFilterCount();
private:
	int filterWidth;
	int filterHeight;
	int filterDepth;
	int stride;
	int pad;
	int filterCount;
};