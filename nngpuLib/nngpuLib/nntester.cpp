#include "nntester.h"
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

NNTester::NNTester()
{

}

NNTester::~NNTester()
{
	if (testingImageData)
	{
		delete testingImageData;
		testingImageData = nullptr;
	}

	if (testingLabelData)
	{
		delete testingLabelData;
		testingLabelData = nullptr;
	}
}

void NNTester::Initialize(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength)
{
	testingImageData = new unsigned char[imageDataLength];
	memcpy(testingImageData, imageData, (size_t)imageDataLength);

	testingLabelData = new unsigned char[labelDataLength];
	memcpy(testingLabelData, labelData, (size_t)labelDataLength);

	testingImageCount = GETINT((testingImageData + NUMBER_OF_ITEMS));
	testingImageWidth = GETINT((testingImageData + NUMBER_OF_COLUMNS));
	testingImageHeight = GETINT((testingImageData + NUMBER_OF_ROWS));

	testingLabelCount = GETINT((testingLabelData + NUMBER_OF_ITEMS));
}

unsigned char* NNTester::GetImage(int imageIndex)
{
	assert(testingImageData);
	assert(imageIndex < testingImageCount);

	return testingImageData + IMAGE_DATA + (testingImageWidth * testingImageHeight * imageIndex);
}

unsigned char NNTester::GetLabel(int labelIndex)
{
	assert(testingLabelData);
	assert(labelIndex < testingLabelCount);

	return *(testingLabelData + LABEL_DATA + labelIndex);
}

void NNTester::Iterate(NNetwork* nn, NNTestResult* testresult)
{
	const int inputCount = testingImageWidth * testingImageHeight;
	const int expectedCount = 10;

	double* input = new double[inputCount];

	unsigned  char* imageData = GetImage(iterationCount);
	for (int index = 0; index < inputCount; index++)
	{
		input[index] = ((double)imageData[index]) / 255;
	}

	nn->Forward(input, inputCount);
	delete input;

	double* output = nullptr;
	int outLength = 0;
	nn->GetOutput(&output, &outLength);

	assert(expectedCount == outLength);
	assert(output);

	double prediction = -1;
	int predictionIndex = -1;
	for (int index = 0; index < expectedCount; index++)
	{
		if (prediction < output[index])
		{
			prediction = output[index];
			predictionIndex = index;
		}
	}

	unsigned char labelData = GetLabel(iterationCount);

	iterationCount++;

	testresult->expected= labelData;
	testresult->predicted = predictionIndex;
}

bool NNTester::TestingComplete()
{
	return iterationCount >= testingImageCount;
}

int NNTester::GetTestingIteration()
{
	return iterationCount;
}