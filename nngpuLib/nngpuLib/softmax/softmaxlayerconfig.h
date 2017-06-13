#pragma once

class SoftmaxLayerConfig
{
public:
	SoftmaxLayerConfig(int size);
	int GetSize();

private:
	int size;
};