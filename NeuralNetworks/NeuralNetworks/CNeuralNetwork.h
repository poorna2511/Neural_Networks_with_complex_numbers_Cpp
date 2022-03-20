#pragma once

#include <iostream>
#include <complex>
#include <vector>

using namespace std;

typedef complex<double> ComplexNum;

class CNeuron
{
public:
	ComplexNum _bias;
	ComplexNum _output;
	ComplexNum _input;
	ComplexNum _error;

	bool _fired;

	int _layerIdx;

	vector<ComplexNum> _weights;

	CNeuron(int iLayerIdx, int iSize, bool iFillRandNums);

	void ComputeOutput();

	void computeError();

};

class CNeuralNetwork
{
private:
	int _numOfLayers;

	double _learningRate;

	// neruons counts in hidden layers
	vector<int> _neuronsCountInEachLayer;
	vector<vector<double>> _posInputData;
	vector<vector<double>> _negInputData;

public:
	CNeuralNetwork(vector<vector<double>> iPosInputData,
		vector<vector<double>> iNegInputData,
		vector<int> iNeuronsCounts,
		double iLearningRate);

	void InitializeNN();

	void StartTraining();

	void ComputeErrors(int iLayerIdx);

	void UpdateWeights(int iLayerIdx);

	void PropagteForwardFrom(int iLayerIdx);

	void PropagteBackWardsTo(int iLayerIdx);
};
