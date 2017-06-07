#pragma once

class ConvLayerConfig
{
public:
	ConvLayerConfig(int filterWidth, int filterHeight, int filterCount, int pad, int stride);

	int GetFilterWidth();
	int GetFilterHeight();
	int GetStride();
	int GetPad();
	int GetFilterCount();
private:
	int filterWidth;
	int filterHeight;
	int stride;
	int pad;
	int filterCount;
};