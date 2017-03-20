#pragma once
#include "nnetwork.h"

class NNTrainer
{
public:
	NNTrainer();
	~NNTrainer();
	void Iterate(NNetwork* nn);
	bool Trainingcomplete();

private:
	int iterationCount = 0;
	int interationMax = 700;


};