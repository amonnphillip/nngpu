#pragma once

class LayerPerfNode
{
private:
	unsigned int timeMs = 0;
	unsigned int bytesTransferred = 0;
	double bytesPerSecond = 0;

public:
	LayerPerfNode() {};
	void Set(unsigned int time, unsigned int bytes);
	void Get(unsigned int& time, double& bytesPerSecond);
};