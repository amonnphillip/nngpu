#pragma once

#include "innetworklayer.h"
#include "fullyconnectedlayerconfig.h"
#include "layer.h"
#include "layerdata.h"
#include "layerperf.h"

class FullyConnectedNode
{
public:
	double bias;
};

class FullyConnectedLayer : public Layer<FullyConnectedNode, double, double, double>, public INNetworkLayer
{
private:
	LayerPerf layerPerf;

protected:
	int nodeCount = 0;
	int forwardCount = 0;
	int backwardWidth = 0;
	int backwardHeight = 0;
	int backwardDepth = 0;
	int weightCount = 0;
	int layerWidth = 0;
	int layerHeight = 0;
	int layerDepth = 0;
	std::unique_ptr<double> weightsHostMem = nullptr;
	double* weightsDeviceMem = nullptr;

public:
	FullyConnectedLayer(FullyConnectedLayerConfig* config, INNetworkLayer* previousLayer);
	virtual void Forward(double* input, int inputSize);
	virtual void Forward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	virtual void Backward(double* input, int inputSize, double learnRate);
	virtual void Backward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
	virtual void Dispose();
	virtual double* GetForwardHostMem(bool copyFromDevice);
	virtual double* GetBackwardHostMem(bool copyFromDevice);
	virtual double* GetForwardDeviceMem();
	virtual double* GetBackwardDeviceMem();
	virtual int GetForwardNodeCount();
	virtual int GetForwardWidth();
	virtual int GetForwardHeight();
	virtual int GetForwardDepth();
	virtual int GetBackwardNodeCount();
	virtual int GetBackwardWidth();
	virtual int GetBackwardHeight();
	virtual int GetBackwardDepth();
	virtual int GetWidth();
	virtual int GetHeight();
	virtual int GetDepth();
	double* GetWeightsForNode(int index);
	double* GetWeightHostMem(bool copyFromDevice);
	int GetWeightCount();
	FullyConnectedNode* GetNodeMem(bool copyFromDevice);
	virtual LayerType GetLayerType();
	virtual void GetLayerData(LayerDataList& layerDataList);
	virtual void GetLayerPerformance(unsigned int& averageTime, double& averageBytes);
	void DebugPrint();
};

