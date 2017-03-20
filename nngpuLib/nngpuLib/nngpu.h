#pragma once
#include "nnetwork.h"
#include "nntrainer.h"
#include "layerdata.h"

class NnGpu
{
public:
	enum LayerDataType {
		Forward = 0,
		Backward = 1
	};

	void InitializeNetwork();
	void AddInputLayer(int width, int height, int depth);
	void AddConvLayer(int filterWidth, int filterHeight, int filterDepth, int filterCount, int pad, int stride);
	void AddReluLayer();
	void AddPoolLayer(int spatialExtent, int stride);
	void AddFullyConnected(int size);
	void AddOutput(int size);
	void GetLayerType(int layerIndex, int* layerType);
	void GetLayerCount(int* layerCount);
	void InitializeTraining();
	bool TrainNetworkInteration();
	void DisposeNetwork();
	void GetLayerDataSize(int layerIndex, int dataType, int* width, int* height, int* depth);
	void GetLayerData(int layerIndex, int dataType, double* layerData);

private:
	NNetwork* nn;
	NNTrainer* trainer;
};
