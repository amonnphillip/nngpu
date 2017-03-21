#include "nngpu.h"
#include "nngpuwin.h"
#include "layerdata.h"

NnGpu* Initialize()
{
	return new NnGpu();
}

void InitializeNetwork(NnGpu* nn)
{
	nn->InitializeNetwork();
}

void AddInputLayer(NnGpu* nn, int width, int height, int depth)
{
	nn->AddInputLayer(width, height, depth);
}

void AddConvLayer(NnGpu* nn, int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride)
{
	nn->AddConvLayer(filterWidth, filterHeight, filterDepth, filterCount, pad, stride);
}

void AddReluLayer(NnGpu* nn)
{
	nn->AddReluLayer();
}

void AddPoolLayer(NnGpu* nn, int spatialExtent, int stride)
{
	nn->AddPoolLayer(spatialExtent, stride);
}

void AddFullyConnected(NnGpu* nn, int size)
{
	nn->AddFullyConnected(size);
}

void AddOutput(NnGpu* nn, int size)
{
	nn->AddOutput(size);
}

void GetLayerType(NnGpu* nn, int layerIndex, int* layerType)
{
	nn->GetLayerType(layerIndex, layerType);
}

void GetLayerCount(NnGpu* nn, int* layerCount)
{
	nn->GetLayerCount(layerCount);
}

void InitializeTraining(NnGpu* nn)
{
	nn->InitializeTraining();
}

bool TrainNetworkInteration(NnGpu* nn)
{
	return nn->TrainNetworkInteration();
}

void GetTrainingIteration(NnGpu* nn, int* interation)
{
	*interation = nn->GetTrainingIteration();
}

void DisposeNetwork(NnGpu* nn)
{
	nn->DisposeNetwork();
}

void GetLayerDataSize(NnGpu* nn, int layerIndex, int dataType, int* width, int* height, int* depth)
{
	nn->GetLayerDataSize(layerIndex, dataType, width, height, depth);
}

void GetLayerData(NnGpu* nn, int layerIndex, int dataType, double* layerData)
{
	nn->GetLayerData(layerIndex, dataType, layerData);
}





