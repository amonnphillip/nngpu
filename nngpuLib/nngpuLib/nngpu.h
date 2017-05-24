#pragma once
#include "nnetwork.h"
#include "nntrainer.h"
#include "nntester.h"
#include "layerdata.h"
#include "nntestresult.h"

class NnGpu
{
public:
	void InitializeNetwork();
	void AddInputLayer(int width, int height, int depth);
	void AddConvLayer(int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride);
	void AddReluLayer();
	void AddPoolLayer(int spatialExtent, int stride);
	void AddFullyConnected(int size);
	void AddOutput(int size);
	void GetLayerType(int layerIndex, int* layerType);
	void GetLayerCount(int* layerCount);
	void InitializeTraining(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength);
	bool TrainNetworkInteration();
	void InitializeTesting(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength);
	bool TestNetworkInteration(NNTestResult* testresult);
	int GetTrainingIteration();
	int GetTestingIteration();
	void DisposeNetwork();
	void GetLayerData(int layerIndex, LayerDataType dataType, LayerDataList& layerData);
	bool RunUnitTests();

private:
	NNetwork* nn;
	NNTrainer* trainer;
	NNTester* tester;
};
