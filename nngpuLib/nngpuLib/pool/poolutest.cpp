#include "poolutest.h"
#include "poolutestreference.h"
#include "testlayer.h"
#include "testutils.h"

bool PoolUTest::Test()
{
	// Set up out test network
	TestLayer* previousLayer = new TestLayer(10, 10, 1);
	previousLayer->ResetForwardAndBackward();

	PoolLayerConfig poolConfig = PoolLayerConfig(1, 1);
	PoolLayer* poolLayer = new PoolLayer(&poolConfig, previousLayer);

	PoolUTestReference* PoolLayerReference = new PoolUTestReference(&poolConfig, previousLayer);

	TestLayer* nextLayer = new TestLayer(10, 10, 1);
	nextLayer->ResetForwardAndBackward();

	// Test forward
	poolLayer->Forward(previousLayer, nextLayer);
	PoolLayerReference->ReferenceForward(previousLayer, nextLayer);

	int errorx;
	int errory;
	int errord;

	const int NUM_OF_TESTS = 3;
	bool* isSame = new bool[NUM_OF_TESTS];
	isSame[0] = TestUtils::CompareRectangularMemory(poolLayer->GetForwardHostMem(true), PoolLayerReference->GetForwardHostMem(true), poolLayer->GetForwardWidth(), poolLayer->GetForwardHeight(), poolLayer->GetForwardDepth(), &errorx, &errory, &errord);
	isSame[1] = TestUtils::CompareMemory(poolLayer->GetBackDataHostMem(true), PoolLayerReference->GetBackDataHostMem(true), poolLayer->GetBackDataNodeCount());

	// Test backward
	poolLayer->Backward(previousLayer, nextLayer, 0.01);
	PoolLayerReference->ReferenceBackward(previousLayer, nextLayer, 0.01);

	isSame[2] = TestUtils::CompareMemory(poolLayer->GetBackwardHostMem(true), PoolLayerReference->GetBackwardHostMem(true), poolLayer->GetBackwardNodeCount());

	bool testResult = TestUtils::AllTrue(isSame, NUM_OF_TESTS);
	delete isSame;

	delete previousLayer;
	delete poolLayer;
	delete PoolLayerReference;
	delete nextLayer;

	return testResult;
}