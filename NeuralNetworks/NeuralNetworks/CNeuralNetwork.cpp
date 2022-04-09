#include "CNeuralNetwork.h"
#include "ActivationFunctions.h"
#include <fstream>
#include <direct.h>

#include <ctime>

#include <filesystem>

//neural networks model with entire neurons data
vector<vector<CNeuron>> NN_MODEL;

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

CNeuralNetwork::CNeuralNetwork(string iStrModelPath)
{
	string strBasicInfofile = iStrModelPath + "BasicInfo.txt";

	LoadBasicInfoOfNN(strBasicInfofile);
	LoadModelDataOfNN(iStrModelPath);
}

void CNeuralNetwork::LoadBasicInfoOfNN(string iStrBasicInfoFilePath)
{
	fstream basicInfofile;
	basicInfofile.open(iStrBasicInfoFilePath.c_str());

	string strLine = "";
	getline(basicInfofile, strLine);
	if (strLine != "learning rate")
		return;

	getline(basicInfofile, strLine);
	_learningRate = stod(strLine);

	getline(basicInfofile, strLine);
	if (strLine != "number of layers")
		return;

	getline(basicInfofile, strLine);
	_numOfLayers = stod(strLine);

	getline(basicInfofile, strLine);
	if (strLine != "neurons count in each layer")
		return;

	NN_MODEL.clear();
	_neuronsCountInEachLayer.clear();

	NN_MODEL.resize(_numOfLayers);
	_neuronsCountInEachLayer.resize(_numOfLayers);
	for (int idx = 0; idx < _numOfLayers; idx++)
	{
		getline(basicInfofile, strLine);
		int neuronCount = stoi(strLine);

		_neuronsCountInEachLayer[idx] = neuronCount;
		NN_MODEL[idx].resize(neuronCount);
	}

	basicInfofile.close();
}

void CNeuralNetwork::LoadModelDataOfNN(string iStrModelPath)
{
	for (int layerIdx = 0; layerIdx < _numOfLayers; layerIdx++)
	{
		vector<CNeuron> &currLayer = NN_MODEL[layerIdx];
		string strLayerFile = iStrModelPath + "\\Layer" + to_string(layerIdx) + ".txt";

		int weightsCount = layerIdx == 0 ? 1 : _neuronsCountInEachLayer[layerIdx - 1];
				
		fstream LayerInfofile;
		LayerInfofile.open(strLayerFile.c_str());

		string strLine = "";
		getline(LayerInfofile, strLine);

		while (LayerInfofile.eof() == false)
		{
			if (strLine.find("neuron") == -1)
				return;

			string str2 = "neuron ";
			string str3 = "";
			strLine.replace(strLine.find(str2), str2.length(), str3);

			int neuronIdx = stoi(strLine);

			CNeuron &neuron = currLayer[neuronIdx];

			getline(LayerInfofile, strLine);
			if (strLine != "bias")
				return;
			
			getline(LayerInfofile, strLine);
			istringstream biasStream(strLine);
			ComplexNum bias;
			biasStream >> bias;
			neuron._bias = bias;

			getline(LayerInfofile, strLine);
			if (strLine != "output")
				return;

			getline(LayerInfofile, strLine);
			istringstream outputStream(strLine);
			ComplexNum output;
			outputStream >> output;
			neuron._output = output;

			getline(LayerInfofile, strLine);
			if (strLine != "input")
				return;

			getline(LayerInfofile, strLine);
			istringstream inputStream(strLine);
			ComplexNum input;
			inputStream >> input;
			neuron._input = input;

			getline(LayerInfofile, strLine);
			if (strLine != "error")
				return;

			getline(LayerInfofile, strLine);
			istringstream errorStream(strLine);
			ComplexNum error;
			errorStream >> error;
			neuron._error = error;

			getline(LayerInfofile, strLine);
			if (strLine != "fired")
				return;

			getline(LayerInfofile, strLine);
			istringstream firedStream(strLine);
			bool fired;
			firedStream >> fired;
			neuron._fired = fired;

			getline(LayerInfofile, strLine);
			if (strLine != "weights")
				return;

			neuron._weights.resize(weightsCount);
			for (int weightIdx = 0; weightIdx < weightsCount; weightIdx++)
			{
				getline(LayerInfofile, strLine);
				istringstream weightStream(strLine);
				ComplexNum weight;
				weightStream >> weight;
				neuron._weights[weightIdx] = weight;
			}

			getline(LayerInfofile, strLine);
		}

		LayerInfofile.close();
	}
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

		SetInputLayerData(inputData);

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

			//SGD
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

void CNeuralNetwork::SaveTheTrainingData()
{
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%d-%m-%Y %H:%M:%S", timeinfo);
	std::string strFolder(buffer);

	//create folder to store the data
	_mkdir(strFolder.c_str());

	// Creation of ofstream class object
	ofstream basicFile;

	string strBasicInfoFile = strFolder + string("\\BasicInfo.txt");
	basicFile.open(strBasicInfoFile.c_str());

	basicFile << "learning rate" << endl << _learningRate << endl;
	basicFile << "number of layers" << endl << _numOfLayers << endl;
	basicFile << "neurons count in each layer" << endl;

	//save neurons count in each layer
	for (int idx = 0; idx < _numOfLayers; idx++)
		basicFile << _neuronsCountInEachLayer[idx] << endl;

	basicFile.close();

	// save model data
	for (int layerIdx = 0; layerIdx < _numOfLayers; layerIdx++)
	{
		string strFile = "\\Layer" + std::to_string(layerIdx);
		string strLayerFile = strFolder + strFile;

		ofstream layerOutput;
		layerOutput.open(strLayerFile.c_str());
		
		vector<CNeuron> &layer = NN_MODEL[layerIdx];
		int neuronCount = _neuronsCountInEachLayer[layerIdx];
		for (int neuronIdx = 0; neuronIdx < neuronCount; neuronIdx++)
		{
			CNeuron &neuron = layer[neuronIdx];

			layerOutput << "neuron " << neuronIdx << endl;
			layerOutput << "bias" << endl << neuron._bias << endl;
			layerOutput << "output" << endl << neuron._output << endl;
			layerOutput << "input" << endl << neuron._input << endl;
			layerOutput << "error" << endl << neuron._error << endl;
			layerOutput << "fired" << endl << neuron._fired << endl;

			layerOutput << "weights" << endl;
			for (int weightIdx = 0; weightIdx < neuron._weights.size(); weightIdx++)
				layerOutput << neuron._weights[weightIdx] << endl;
		}

		layerOutput.close();
	}



	//filesystcreate_directory("sandbox/1/2/b");


}

void CNeuralNetwork::SaveNeuralNetwork()
{
	ofstream nnFile;
	nnFile.open("NN.txt", ios::app);
	
	nnFile.write((char*)&_numOfLayers, sizeof(_numOfLayers));

	nnFile.write((char*)&_learningRate, sizeof(_learningRate));

	nnFile.write((char*)&_neuronsCountInEachLayer, sizeof(_neuronsCountInEachLayer));

	nnFile.write((char*)&NN_MODEL, sizeof(NN_MODEL));

	nnFile.close();
}

void CNeuralNetwork::LoadNeuralNetwork(string iStrFilePath)
{
	ifstream nnFile;

	nnFile.open(iStrFilePath.c_str(), ios::in);

	nnFile.read((char*)&_numOfLayers, sizeof(_numOfLayers));

	nnFile.read((char*)&_learningRate, sizeof(_learningRate));

	_neuronsCountInEachLayer.clear();
	nnFile.read((char*)&_neuronsCountInEachLayer, sizeof(_neuronsCountInEachLayer));

	NN_MODEL.clear();
	nnFile.read((char*)&NN_MODEL, sizeof(NN_MODEL));

	nnFile.close();

}

void CNeuralNetwork::SetInputLayerData(const vector<double> &iInputData)
{
	vector<CNeuron> &inputLayer = NN_MODEL[0]; //input layer

	for (int inputIdx = 0; inputIdx < iInputData.size(); inputIdx++)
	{
		CNeuron &neuron = inputLayer[inputIdx];
		neuron._input = iInputData[inputIdx];
	}
}
