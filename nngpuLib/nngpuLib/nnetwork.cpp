#include <iterator>
#include "nnetwork.h"
#include "inputlayer.h"

void NNetwork::Forward(double* input, int inputSize)
{
	INNetworkLayer* previousLayer = nullptr;
	INNetworkLayer* nextLayer = nullptr;
	for (int index = 0; index < layers.size(); index++)
	{
		INNetworkLayer* layer = layers.at(index);
		if (index + 1 < layers.size())
		{
			nextLayer = layers.at(index + 1);
		} 
		else
		{
			nextLayer = nullptr;
		}

		if (index == 0)
		{
			layer->Forward(input, inputSize);
		}
		else
		{
			layer->Forward(previousLayer, nextLayer);
		}

		previousLayer = layer;
	}
}

void NNetwork::Backward(double* expected, int expectedSize, double learnRate)
{
	INNetworkLayer* nextLayer = nullptr;
	for (int index = (int)(layers.size() - 1); index > 0; index--)
	{
		INNetworkLayer* layer = layers.at(index);
		INNetworkLayer* previousLayer = layers.at(index - 1);

		if (index == layers.size() - 1)
		{
			layer->Backward(expected, expectedSize, learnRate);
		}
		else
		{
			layer->Backward(previousLayer, nextLayer, learnRate);
		}

		nextLayer = layer;
	}
}

void NNetwork::GetOutput(double** output, int* outputLength)
{
	int lastLayer = (int)layers.size() - 1;
	INNetworkLayer* layer = layers.at(lastLayer);

	*output = layers.at(lastLayer)->GetForwardHostMem(true);
	*outputLength = layers.at(lastLayer)->GetForwardNodeCount();
}

double* NNetwork::GetLayerForward(int layerIndex)
{
	return layers.at(layerIndex)->GetForwardHostMem(false);
}

double* NNetwork::GetLayerBackward(int layerIndex)
{
	return layers.at(layerIndex)->GetBackwardHostMem(false);
}

void NNetwork::Dispose()
{
	while (!layers.empty())
	{
		layers.pop_back();
	}
}

size_t NNetwork::GetLayerCount()
{
	return layers.size();
}