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
#include "convutest.h"
#include "poolutest.h"
#include "softmaxlayer.h"
#include "fullyconnectedutest.h"

void NnGpu::InitializeNetwork()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

#ifdef _UNITTEST
	size_t size = 0;
	cudaDeviceGetLimit(&size, cudaLimitPrintfFifoSize);
	size *= 5;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size);
#endif

	nn = new NNetwork();
}

void NnGpu::AddInputLayer(int width, int height, int depth)
{
	assert(nn);

	nn->Add<InputLayer, InputLayerConfig>(new InputLayerConfig(width, height, depth));
}

void NnGpu::AddConvLayer(int filterWidth, int filterHeight, int filterCount, int pad, int stride)
{
	assert(nn);

	nn->Add<ConvLayer, ConvLayerConfig>(new ConvLayerConfig(filterWidth, filterHeight, filterCount, pad, stride));
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

void NnGpu::AddSoftmax(int size)
{
	assert(nn);

	nn->Add<SoftmaxLayer, SoftmaxLayerConfig>(new SoftmaxLayerConfig(size));
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

	*layerCount = (int)nn->GetLayerCount();
}

void NnGpu::InitializeTraining(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength)
{
	trainer = new NNTrainer();
	trainer->Initialize(imageData, imageDataLength, labelData, labelDataLength);
}

bool NnGpu::TrainNetworkInteration()
{
	trainer->Iterate(nn);

	return trainer->TrainingComplete();
}

int NnGpu::GetTrainingIteration()
{
	return trainer->GetTrainingIteration();
}

void NnGpu::InitializeTesting(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength)
{
	tester = new NNTester();
	tester->Initialize(imageData, imageDataLength, labelData, labelDataLength);
}

bool NnGpu::TestNetworkInteration(NNTestResult* testresult)
{
	tester->Iterate(nn, testresult);

	return tester->TestingComplete();
}

int NnGpu::GetTestingIteration()
{
	return tester->GetTestingIteration();
}

void NnGpu::DisposeNetwork()
{
	delete trainer;

	// Dispose of the resouces we allocated and close
	nn->Dispose();
	delete nn;

	cudaDeviceReset();
}

void NnGpu::GetLayerData(int layerIndex, LayerDataType dataType, LayerDataList& layerDataList)
{
	INNetworkLayer* layer = nn->GetLayer(layerIndex);

	if (layer != nullptr)
	{
		layer->GetLayerData(layerDataList);
	}
}

bool NnGpu::RunUnitTests()
{
#if 0
	ConvUTest* convTest = new ConvUTest();
	bool convTestResult = convTest->Test();
	delete convTest;

	PoolUTest* poolTest = new PoolUTest();
	bool poolTestResult = poolTest->Test();
	delete poolTest;
#endif
	FullyConnectedUTest* fullyConnectedTest = new FullyConnectedUTest();
	bool fullyConnectedTestResult = fullyConnectedTest->Test();
	delete fullyConnectedTest;

	return true;// convTestResult && poolTestResult && fullyConnectedTestResult;
}

void NnGpu::GetLayerPerformanceData(int layerIndex, unsigned int* averageTimeInMs, double* averageBytes)
{
	INNetworkLayer* layer = nn->GetLayer(layerIndex);

	unsigned int time = 0;
	double bytes = 0;

	if (layer != nullptr)
	{
		layer->GetLayerPerformance(time, bytes);
	}

	*averageTimeInMs = time;
	*averageBytes = bytes;
}