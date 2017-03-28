#include "nntrainer.h"
#include "inputlayer.h"
#include "inputlayerconfig.h"
#include "fullyconnectedlayer.h"
#include "relulayer.h"
#include "poollayer.h"
#include "convlayer.h"
#include "outputlayer.h"

NNTrainer::NNTrainer()
{

}

NNTrainer::~NNTrainer()
{

}

void NNTrainer::Iterate(NNetwork* nn)
{
	const int inputCount = 64;
	const int expectedCount = 2;
	double* input;
	double* expected;

	if (iterationCount & 1)
	{
		double inputAlt[] = {
			1, 1,  0, 0,  0, 0, 0, 0,
			1, 1,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
		};
		input = inputAlt;

		double expectedAlt[] = { 1, 0 };
		expected = expectedAlt;
	}
	else
	{
		double inputAlt[] = {
			0, 0,  1, 1,  0, 0, 0, 0,
			0, 0,  1, 1,  0, 0, 0, 0,
			1, 1,  0, 0,  0, 0, 0, 0,
			1, 1,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
			0, 0,  0, 0,  0, 0, 0, 0,
		};
		input = inputAlt;

		double expectedAlt[] = { 0, 1 };
		expected = expectedAlt;
	}


	nn->Forward(input, inputCount);
	nn->Backward(expected, expectedCount, 0.001);

	iterationCount++;
}

bool NNTrainer::Trainingcomplete()
{
	return iterationCount >= interationMax;
}

int NNTrainer::GetTrainingIteration()
{
	return iterationCount;
}