#pragma once

#include <iostream>
#include <complex>

using namespace std;

typedef complex<double> ComplexNum;

class CActivationFunction {

public:
	CActivationFunction() {}

	static bool ActivationFunctions(ComplexNum iInput, ComplexNum &oOutput);
	static ComplexNum DerivativeActivationFunctions(ComplexNum iInput);
	
	static bool FastSigmoid(ComplexNum iInput, ComplexNum &oOutput);

	static bool Tanh(ComplexNum iInput, ComplexNum &oOutput);
	static ComplexNum DerivativeTanh(ComplexNum iInput);
};
