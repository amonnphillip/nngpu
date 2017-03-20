#include "nngpu.h"
#include "nngpuwin.h"

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

void InitializeTraining(NnGpu* nn)
{
	nn->InitializeTraining();
}

bool TrainNetworkInteration(NnGpu* nn)
{
	return nn->TrainNetworkInteration();
}

void DisposeNetwork(NnGpu* nn)
{
	nn->DisposeNetwork();
}






