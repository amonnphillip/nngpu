
#include "fullyconnectedutest.h"
#include "fullyconnectedutestreference.h"
#include "testlayer.h"
#include "testutils.h"

bool FullyConnectedUTest::Test()
{
	const int LAYER_WIDTH = 7;
	const int LAYER_HEIGHT = 7;
	const int LAYER_DEPTH = 32;

	// Set up out test network
	TestLayer* previousLayer = new TestLayer(LAYER_WIDTH, LAYER_HEIGHT, LAYER_DEPTH);
	previousLayer->ResetForwardAndBackward();

	FullyConnectedLayerConfig config = FullyConnectedLayerConfig(10, 1, 1);
	FullyConnectedLayer* fullyConnectedLayer = new FullyConnectedLayer(&config, previousLayer);

	FullyConnectedUTestReference* fullyConnectedLayerReference = new FullyConnectedUTestReference(&config, previousLayer);

	TestLayer* nextLayer = new TestLayer(10, 1, 1);
	nextLayer->ResetForwardAndBackward();

	// Test forward
	fullyConnectedLayer->Forward(previousLayer, nextLayer);
	fullyConnectedLayerReference->ReferenceForward(previousLayer, nextLayer);

	int errorx;
	int errory;
	int errord;

	const int NUM_OF_TESTS = 4;
	bool* isSame = new bool[NUM_OF_TESTS];

	isSame[0] = TestUtils::CompareRectangularMemory(fullyConnectedLayer->GetForwardHostMem(true), fullyConnectedLayerReference->GetForwardHostMem(true), fullyConnectedLayer->GetForwardWidth(), fullyConnectedLayer->GetForwardHeight(), fullyConnectedLayer->GetForwardDepth(), &errorx, &errory, &errord);

	// Test backward
	fullyConnectedLayer->Backward(previousLayer, nextLayer, 0.01);
	fullyConnectedLayerReference->ReferenceBackward(previousLayer, nextLayer, 0.01);

	isSame[1] = TestUtils::CompareMemory(fullyConnectedLayer->GetWeightHostMem(true), fullyConnectedLayerReference->GetWeightHostMem(true), fullyConnectedLayer->GetWeightCount() * fullyConnectedLayer->GetForwardNodeCount());
	isSame[2] = TestUtils::CompareRectangularMemory(fullyConnectedLayer->GetBackwardHostMem(true), fullyConnectedLayerReference->GetBackwardHostMem(true), fullyConnectedLayer->GetBackwardWidth(), fullyConnectedLayer->GetBackwardHeight(), fullyConnectedLayer->GetBackwardDepth(), &errorx, &errory, &errord);

	FullyConnectedNode* a = fullyConnectedLayer->GetNodeMem(true);
	FullyConnectedNode* b = fullyConnectedLayerReference->GetNodeMem(true);
	bool biasSame = true;
	for (int index = 0; index < fullyConnectedLayer->GetForwardNodeCount(); index++)
	{
		if (a->bias != b->bias)
		{
			biasSame = false;
		}
		a++;
		b++;
	}
	isSame[3] = biasSame;

	bool testResult = TestUtils::AllTrue(isSame, NUM_OF_TESTS);
	delete isSame;

	delete previousLayer;
	delete fullyConnectedLayer;
	delete fullyConnectedLayerReference;
	delete nextLayer;

	return testResult;
}