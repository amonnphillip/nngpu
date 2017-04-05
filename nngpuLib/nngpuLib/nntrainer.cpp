#include "nntrainer.h"
#include "inputlayer.h"
#include "inputlayerconfig.h"
#include "fullyconnectedlayer.h"
#include "relulayer.h"
#include "poollayer.h"
#include "convlayer.h"
#include "outputlayer.h"


#define MAGIC_NUMBER 0
#define NUMBER_OF_ITEMS 4
#define NUMBER_OF_ROWS 8
#define NUMBER_OF_COLUMNS 12
#define IMAGE_DATA 16

#define MAGIC_NUMBER 0
#define NUMBER_OF_ITEMS 4
#define LABEL_DATA 8


#define GETINT(a) (int)((((unsigned int)a[0]) << 24) + (((unsigned int)a[1]) << 16) + (((unsigned int)a[2]) << 8) + ((unsigned int)(a[3])))

NNTrainer::NNTrainer()
{

}

NNTrainer::~NNTrainer()
{
	if (trainingImageData)
	{
		delete trainingImageData;
		trainingImageData = nullptr;
	}

	if (trainingLabelData)
	{
		delete trainingLabelData;
		trainingLabelData = nullptr;
	}
}

void NNTrainer::Initialize(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength)
{
	trainingImageData = new unsigned char[imageDataLength];
	memcpy(trainingImageData, imageData, (size_t)imageDataLength);

	trainingLabelData = new unsigned char[labelDataLength];
	memcpy(trainingLabelData, labelData, (size_t)labelDataLength);

	trainingImageCount = GETINT((trainingImageData + NUMBER_OF_ITEMS));
	trainingImageWidth = GETINT((trainingImageData + NUMBER_OF_COLUMNS));
	trainingImageHeight = GETINT((trainingImageData + NUMBER_OF_ROWS));

	trainingLabelCount = GETINT((trainingLabelData + NUMBER_OF_ITEMS));
}

unsigned char* NNTrainer::GetImage(int imageIndex)
{
	assert(trainingImageData);
	assert(imageIndex < trainingImageCount);

	return trainingImageData + IMAGE_DATA + (trainingImageWidth * trainingImageHeight * imageIndex);
}

unsigned char NNTrainer::GetLabel(int labelIndex)
{
	assert(trainingLabelData);
	assert(labelIndex < trainingLabelCount);

	return *(trainingLabelData + LABEL_DATA + labelIndex);
}

void NNTrainer::Iterate(NNetwork* nn)
{
	const int inputCount = trainingImageWidth * trainingImageHeight;
	const int expectedCount = 10;

	double* input = new double[inputCount];
	double* expected = new double[expectedCount];

	unsigned  char* imageData = GetImage(iterationCount);
	for (int index = 0; index < inputCount; index++)
	{
		input[index] = ((double)imageData[index]) / 255;
	}

	unsigned char labelData = GetLabel(iterationCount);
	for (int index = 0; index < expectedCount; index++)
	{
		if (labelData == index)
		{
			expected[index] = 1;
		}
		else
		{
			expected[index] = 0;
		}
	}


	nn->Forward(input, inputCount);
	nn->Backward(expected, expectedCount, 0.001);

	delete input;
	delete expected;

	iterationCount++;
}

bool NNTrainer::TrainingComplete()
{
	return iterationCount >= trainingImageCount;
}

int NNTrainer::GetTrainingIteration()
{
	return iterationCount;
}