#pragma once
#include "nnetwork.h"

class NNTrainer
{
public:
	NNTrainer();
	~NNTrainer();
	void Initialize(unsigned char* imageData, int imageDataLength, unsigned char* labelData, int labelDataLength, int epocMax);
	void Iterate(NNetwork* nn);
	bool TrainingComplete();
	int GetTrainingIteration();
	unsigned char* GetImage(int imageIndex);
	unsigned char GetLabel(int labelIndex);

private:
	int iterationCount = 0;

	unsigned char* trainingImageData = nullptr;
	unsigned char* trainingLabelData = nullptr;
	int trainingEpocCount;
	int trainingImageCount;
	int trainingImageWidth;
	int trainingImageHeight;
	int trainingLabelCount;
	bool doneTraining;
};