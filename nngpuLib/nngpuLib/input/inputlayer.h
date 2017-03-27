#pragma once

#include "inputlayerconfig.h"
#include "innetworklayer.h"
#include "layer.h"
#include "layerdata.h"

class InputNode
{
};

class InputLayer : public Layer<InputNode, double, double, double>, public INNetworkLayer
{
private:
	int width;
	int height;
	int depth;
	int nodeCount = 0;

public:
	InputLayer(InputLayerConfig* config, INNetworkLayer* previousLayer);
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
	virtual int GetWidth();
	virtual int GetHeight();
	virtual int GetDepth();
	virtual LayerType GetLayerType();
	virtual void GetLayerData(LayerDataList& layerDataList);
	void DebugPrint();
};
