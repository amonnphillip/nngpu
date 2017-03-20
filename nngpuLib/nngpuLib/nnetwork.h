#pragma once
#include "innetworklayer.h"
#include <cassert>
#include <vector>

class NNetwork
{
private:
	std::vector<INNetworkLayer*> layers;

public:
	template<class layertype, class layerConfigType> 
	void Add(layerConfigType* config)
	{
		INNetworkLayer* previousLayer = nullptr;
		if (layers.size() > 0)
		{
			previousLayer = layers.at(layers.size() - 1);
		}

		layertype* layer = new layertype(config, previousLayer);
		layers.push_back(layer);
	}
	template<class layertype>
	void Add()
	{
		INNetworkLayer* previousLayer = nullptr;
		if (layers.size() > 0)
		{
			previousLayer = layers.at(layers.size() - 1);
		}

		layertype* layer = new layertype(previousLayer);
		layers.push_back(layer);
	}
	void Forward(double* input, int inputSize);
	void Backward(double* input, int expectedSize, double learnRate);
	double* GetLayerForward(int layerIndex);
	double* GetLayerBackward(int layerIndex);
	void Dispose();
	size_t GetLayerCount();
	INNetworkLayer* GetLayer(int layerIndex)
	{
		return layers.at(layerIndex);
	}
};
