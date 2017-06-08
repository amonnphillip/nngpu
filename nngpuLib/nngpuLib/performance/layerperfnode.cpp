#include "layerperfnode.h"


void LayerPerfNode::Set(unsigned int time, unsigned int bytes)
{
	timeMs = time;
	bytesTransferred = bytes;
	double timeS = (double)timeMs / 1000.0;
	bytesPerSecond = ((double)bytesTransferred / timeS);
}

void LayerPerfNode::Get(unsigned int& time, double& bytes)
{
	time = timeMs;
	bytes = bytesPerSecond;
}