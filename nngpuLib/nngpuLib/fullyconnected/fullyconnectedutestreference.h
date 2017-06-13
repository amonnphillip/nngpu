#pragma once

#include "fullyconnectedlayer.h"

class FullyConnectedUTestReference : public FullyConnectedLayer
{
public:
	FullyConnectedUTestReference(FullyConnectedLayerConfig* config, INNetworkLayer* previousLayer);
	void ReferenceForward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	void ReferenceBackward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
};