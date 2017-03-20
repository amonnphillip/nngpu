#include <iostream>
#include "cuda_runtime.h"
#include "nngpu.h"
#include "nnetwork.h"
#include "inputlayer.h"
#include "inputlayerconfig.h"
#include "fullyconnectedlayer.h"
#include "relulayer.h"
#include "poollayer.h"
#include "convlayer.h"
#include "outputlayer.h"
#include "nntrainer.h"
#include "layerdata.h"

void NnGpu::InitializeNetwork()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	nn = new NNetwork();

/*
	// Create the (very small) network
	nn = new NNetwork();
	nn->Add<InputLayer, InputLayerConfig>(new InputLayerConfig(8, 8, 1));
	//nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(8, 8, 1));
	nn->Add<ConvLayer, ConvLayerConfig>(new ConvLayerConfig(3, 3, 1, 4, 1, 1));
	nn->Add<ReluLayer>();
	nn->Add<PoolLayer, PoolLayerConfig>(new PoolLayerConfig(2, 2));
	//nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(2));
	nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(2));
	nn->Add<OutputLayer, OutputLayerConfig>(new OutputLayerConfig(2));*/
}

void NnGpu::AddInputLayer(int width, int height, int depth)
{
	assert(nn);

	nn->Add<InputLayer, InputLayerConfig>(new InputLayerConfig(width, height, depth));
}

void NnGpu::AddConvLayer(int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride)
{
	assert(nn);

	nn->Add<ConvLayer, ConvLayerConfig>(new ConvLayerConfig(filterWidth, filterHeight, filterDepth, filterCount, pad, stride));
}

void NnGpu::AddReluLayer()
{
	assert(nn);

	nn->Add<ReluLayer>();
}

void NnGpu::AddPoolLayer(int spatialExtent, int stride)
{
	assert(nn);

	nn->Add<PoolLayer, PoolLayerConfig>(new PoolLayerConfig(spatialExtent, stride));
}

void NnGpu::AddFullyConnected(int size)
{
	assert(nn);

	nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(size));
}

void NnGpu::AddOutput(int size)
{
	assert(nn);

	nn->Add<OutputLayer, OutputLayerConfig>(new OutputLayerConfig(size));
}

void NnGpu::GetLayerType(int layerIndex, int* layerType)
{
	assert(nn);
	assert(layerType);

	INNetworkLayer* layer = nn->GetLayer(layerIndex);
	*layerType = layer->GetLayerType();
}

void NnGpu::GetLayerCount(int* layerCount)
{
	assert(layerCount);

	*layerCount = nn->GetLayerCount();
}

void NnGpu::InitializeTraining()
{
	trainer = new NNTrainer();
}

bool NnGpu::TrainNetworkInteration()
{
	trainer->Iterate(nn);

	return trainer->Trainingcomplete();
}

void NnGpu::DisposeNetwork()
{
	delete trainer;

	// Dispose of the resouces we allocated and close
	nn->Dispose();
	delete nn;

	cudaDeviceReset();
}

void NnGpu::GetLayerDataSize(int layerIndex, int dataType, int* width, int* height, int* depth)
{
	INNetworkLayer* layer = nn->GetLayer(layerIndex);

	if (layer != nullptr)
	{
		if (dataType == LayerDataType::Forward)
		{
			*width = layer->GetForwardWidth();
			*height = layer->GetForwardHeight();
			*depth = layer->GetForwardDepth();
		}
	}
}

void NnGpu::GetLayerData(int layerIndex, int dataType, double* layerData)
{
	INNetworkLayer* layer = nn->GetLayer(layerIndex);

	if (layer != nullptr)
	{
		if (dataType == LayerDataType::Forward)
		{
			int size = layer->GetForwardNodeCount();
			double* layerHostMem = layer->GetForwardHostMem(true);

			memcpy(layerData, layerHostMem, (size_t)size * sizeof(double));
		}
	}
}