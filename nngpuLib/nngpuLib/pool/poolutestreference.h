#pragma once

#include "poollayer.h"

class PoolUTestReference : public PoolLayer
{
public:
	PoolUTestReference(PoolLayerConfig* config, INNetworkLayer* previousLayer);
	void ReferenceForward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer);
	void ReferenceBackward(INNetworkLayer* previousLayer, INNetworkLayer* nextLayer, double learnRate);
};