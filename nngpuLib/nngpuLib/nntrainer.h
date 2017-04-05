#pragma once
#include "nnetwork.h"

class NNTrainer
{
public:
	NNTrainer();
	~NNTrainer();
	void Initialize(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength);
	void Iterate(NNetwork* nn);
	bool TrainingComplete();
	int GetTrainingIteration();
	unsigned char* GetImage(int imageIndex);
	unsigned char GetLabel(int labelIndex);

private:
	int iterationCount = 0;

	unsigned char* trainingImageData;
	unsigned char* trainingLabelData;
	int trainingImageCount;
	int trainingImageWidth;
	int trainingImageHeight;
	int trainingLabelCount;
};