#pragma once
#include "nnetwork.h"
#include "nntrainer.h"
#include "layerdata.h"

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
	void GetLayerData2(int layerIndex, int dataType, int dataIndex, int* count);
	void GetLayerType(int layerIndex, int* layerType);
	void GetLayerCount(int* layerCount);
	void InitializeTraining();
	bool TrainNetworkInteration();
	int GetTrainingIteration();
	void DisposeNetwork();
	void GetLayerData(int layerIndex, LayerDataType dataType, LayerDataList& layerData);

private:
	NNetwork* nn;
	NNTrainer* trainer;
};
