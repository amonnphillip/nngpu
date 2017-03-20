#include <iostream>
#include "cuda_runtime.h"
#include "nnetwork.h"
#include "inputlayer.h"
#include "inputlayerconfig.h"
#include "fullyconnectedlayer.h"
#include "relulayer.h"
#include "poollayer.h"
#include "convlayer.h"
#include "outputlayer.h"

int main()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	// Create the (very small) network
	NNetwork* nn = new NNetwork();
	nn->Add<InputLayer, InputLayerConfig>(new InputLayerConfig(8, 8, 1));
	//nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(8, 8, 1));
	nn->Add<ConvLayer, ConvLayerConfig>(new ConvLayerConfig(3, 3, 1, 4, 1, 1));
	nn->Add<ReluLayer>();
	nn->Add<PoolLayer, PoolLayerConfig>(new PoolLayerConfig(2, 2));
	//nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(2));
	nn->Add<FullyConnectedLayer, FullyConnectedLayerConfig>(new FullyConnectedLayerConfig(2));
	nn->Add<OutputLayer, OutputLayerConfig>(new OutputLayerConfig(2));

	// Train the network
	int iterationCount = 0;
	int interationMax = 700;
	while (iterationCount < interationMax)
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
		nn->Backward(expected, expectedCount, 0.01);


		// Display some layers in the console
		std::cout << "iteration: " << iterationCount << "\r\n";
		for (int layerIndex = 0; layerIndex < nn->GetLayerCount(); layerIndex++)
		{
			INNetworkLayer* layer = nn->GetLayer(layerIndex);
			LayerType layerType = layer->GetLayerType();
			if (layerType == LayerType::Input)
			{
				InputLayer* prntLayer = dynamic_cast<InputLayer*>(layer);
				prntLayer->DebugPrint();
			}
			else if (layerType == LayerType::Convolution)
			{
				ConvLayer* prntLayer = dynamic_cast<ConvLayer*>(layer);
				prntLayer->DebugPrint();
			}
			else if (layerType == LayerType::FullyConnected)
			{
				FullyConnectedLayer* prntLayer = dynamic_cast<FullyConnectedLayer*>(layer);
				prntLayer->DebugPrint();
			}
			else if (layerType == LayerType::Output)
			{
				OutputLayer* prntLayer = dynamic_cast<OutputLayer*>(layer);
				prntLayer->DebugPrint(expected, expectedCount);
			}

			std::cout << "\r\n";
		}

		std::cout << "\r\n\r\n";

		iterationCount++;
	}
	
	// Dispose of the resouces we allocated and close
	nn->Dispose();
	delete nn;

	cudaDeviceReset();
}

