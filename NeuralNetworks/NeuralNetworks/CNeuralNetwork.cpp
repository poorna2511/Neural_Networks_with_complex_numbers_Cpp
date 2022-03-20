#include "CNeuralNetwork.h"
#include "ActivationFunctions.h"

//neural networks model with entire neurons data
vector < vector<CNeuron>> NN_MODEL;

double randd() {
	return (double)rand() / ((double)RAND_MAX + 1);
}

CNeuron::CNeuron(int iLayerIdx, int iSize, bool iFillRandNums)
{
	_layerIdx = iLayerIdx;
	_bias = ComplexNum(1, 0);//iFillRandNums == true ? ComplexNum(rand(), rand()) : ComplexNum(0, 0);//random bias

	_weights.reserve(iSize);
	for (int idx = 0; idx < iSize; idx++)
		_weights[idx] = iFillRandNums == true ? ComplexNum(randd(), randd()) : ComplexNum(0, 0);//random weight
}

void CNeuron::ComputeOutput()
{
	int prevLayerIdx = _layerIdx - 1;

	if (prevLayerIdx >= 1)//if not an input layer
	{
		int neuronsCount = NN_MODEL[prevLayerIdx].size();// for previous layer neuron count

		ComplexNum sigmaXiWi = 0;
		for (int neuronIdx = 0; neuronIdx < neuronsCount; neuronIdx++)
		{
			CNeuron neuron = NN_MODEL[prevLayerIdx][neuronIdx];

			ComplexNum x = neuron._output;
			ComplexNum weight = _weights[neuronIdx];

			sigmaXiWi = sigmaXiWi + x * weight;
		}

		_input = sigmaXiWi + _bias;
		_fired = CActivationFunction::ActivationFunctions(_input, _output);
	}
	else
	{	//if neuron from input layer
		ComplexNum input = _input + _bias;
		_fired = CActivationFunction::ActivationFunctions(input, _output);
	}
}

void CNeuron::computeError()
{
	int layerCount = NN_MODEL.size();
	if (_layerIdx == layerCount - 1)// for output layer
	{
		_error = ComplexNum(1, 0) - _output;
	}
	else
	{
		_error = 0;

		vector<CNeuron> &nextLayer = NN_MODEL[_layerIdx + 1];
		for (int neuronIdx = 0; neuronIdx < nextLayer.size(); neuronIdx++)
		{
			CNeuron &neuron = nextLayer[neuronIdx];
			ComplexNum weight = neuron._weights[_layerIdx];
			ComplexNum error = neuron._error;
			_error += weight * error;
		}
	}
}

CNeuralNetwork::CNeuralNetwork(vector<vector<double>> iPosInputData,
	vector<vector<double>> iNegInputData,
	vector<int> iNeuronsCounts,
	double iLearningRate)
{
	_learningRate = iLearningRate;
	_neuronsCountInEachLayer = iNeuronsCounts;
	_posInputData = iPosInputData;
	_negInputData = iNegInputData;

	InitializeNN();
}

void CNeuralNetwork::InitializeNN()
{
	auto startItr = _neuronsCountInEachLayer.begin();
	_neuronsCountInEachLayer.insert(startItr, _posInputData[0].size()); //first layer with neurons of input data
	_neuronsCountInEachLayer.push_back(1); //last layer with one neuron i.e., output

	_numOfLayers = _neuronsCountInEachLayer.size();//input + hidden + output layer

	//assign input data to the first layer
	NN_MODEL.clear();
	NN_MODEL.reserve(_numOfLayers);
	for (int layerIdx = 0; layerIdx < _numOfLayers; layerIdx++)
	{
		int neuronCount = _neuronsCountInEachLayer[layerIdx];

		vector<CNeuron> &layer = NN_MODEL[layerIdx];
		layer.clear();
		layer.reserve(neuronCount);
		for (int neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
		{
			CNeuron &neuron = layer[neuronIdx];
			int size = layerIdx == 0 ? 1 : _neuronsCountInEachLayer[layerIdx - 1];
			neuron = CNeuron(layerIdx, size, true);

			if (layerIdx == 0)//first layer is input
				neuron._input = _posInputData[0][neuronIdx];//take first positive data and initialize

			//if (layerIdx == _numOfLayers - 1)//last layer will have no bias
			//	neuron._bias = 0;
		}
	}
}

void CNeuralNetwork::StartTraining()
{
	for (int posIdx = 0; posIdx < _posInputData.size(); posIdx++)
	{
		const vector<double> &inputData = _posInputData[posIdx];
		vector<CNeuron> &inputLayer = NN_MODEL[0]; //input layer

		for (int inputIdx = 0; inputIdx < inputData.size(); inputIdx++)
		{
			CNeuron &neuron = inputLayer[inputIdx];
			neuron._input = inputData[inputIdx];
		}

		//perform training for each layer in neural net other than input layer
		for (int currLayIdx = 1; currLayIdx < _numOfLayers; currLayIdx++)
		{
			for (int itr = 0; itr < 4; itr++)// we iterate 4 times for each layer training
			{
				PropagteForwardFrom(currLayIdx);

				ComputeErrors(currLayIdx);
				UpdateWeights(currLayIdx);
			}
		}
	}

	cout << "Training completed" << endl;
	cout << "Output = " << NN_MODEL[_numOfLayers - 1][0]._error;
}

void CNeuralNetwork::ComputeErrors(int iLayerIdx)
{
	//start calculating errors from output layer to iLayerIdx in back direction
	for (int layerIdx = _numOfLayers - 1; layerIdx >= iLayerIdx; layerIdx--)
	{
		vector<CNeuron> &layer = NN_MODEL[layerIdx];
		for (int neuronIdx = 0; neuronIdx < layer.size(); neuronIdx++)
		{
			CNeuron &neuron = layer[neuronIdx];
			neuron.computeError();
		}
	}
}

void CNeuralNetwork::UpdateWeights(int iLayerIdx)
{
	//updates weights of neurons in current layer according to the error
	vector<CNeuron> &currLayer = NN_MODEL[iLayerIdx];
	for (int neuronIdx = 0; neuronIdx < currLayer.size(); neuronIdx++)
	{
		CNeuron &neuron = currLayer[neuronIdx];
		vector<ComplexNum> &weights = neuron._weights;
		for (int weightIdx = 0; weightIdx < weights.size(); weightIdx++)
		{
			ComplexNum deriv = CActivationFunction::DerivativeActivationFunctions(neuron._input);
			ComplexNum error = neuron._error;
			ComplexNum output = neuron._output;

			weights[weightIdx] += (_learningRate*error*deriv*output);
		}
	}
}

void CNeuralNetwork::PropagteForwardFrom(int iFromLayerIdx)
{
	//update values neuron by neuron and layer by layer
	//updating from current layer to output layer
	for (int layIdx = iFromLayerIdx; layIdx < _numOfLayers; layIdx++)
	{
		vector<CNeuron> &layer = NN_MODEL[layIdx];
		int neuronCount = layer.size();

		for (int neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
		{
			CNeuron &neuron = layer[neuronIdx];
			neuron.ComputeOutput();
		}
	}
}

void CNeuralNetwork::PropagteBackWardsTo(int iToLayerIdx)
{

}