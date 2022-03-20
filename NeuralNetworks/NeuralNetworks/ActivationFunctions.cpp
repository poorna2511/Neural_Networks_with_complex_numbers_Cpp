#include "ActivationFunctions.h"

bool CActivationFunction::ActivationFunctions(ComplexNum iInput, ComplexNum &oOutput)
{
	return Tanh(iInput, oOutput);;
}

ComplexNum CActivationFunction::DerivativeActivationFunctions(ComplexNum iInput)
{
	return DerivativeTanh(iInput);
}

bool CActivationFunction::FastSigmoid(ComplexNum iInput, ComplexNum &oOutput)
{
	double real = iInput.real();
	double img = iInput.imag();

	double realOut = real / (1 + abs(real));
	realOut = (realOut + 1) / 2;

	double imgOut = img / (1 + abs(img));
	imgOut = (imgOut + 1) / 2;

	oOutput.real(realOut);
	oOutput.imag(imgOut);

	return realOut > 0.5 ? true : false;
}

bool CActivationFunction::Tanh(ComplexNum iInput, ComplexNum &oOutput)
{
	double real = iInput.real();
	double img = iInput.imag();

	double realOut = tanh(real);
	double imgOut = tanh(img);

	oOutput.real(realOut);
	oOutput.imag(imgOut);

	return realOut > 0 ? true : false;
}

ComplexNum CActivationFunction::DerivativeTanh(ComplexNum iInput)
{
	double real = iInput.real();
	double img = iInput.imag();

	double realOut = 1 - (tanh(real)*tanh(real));
	double imgOut = 1- (tanh(img)*tanh(img));

	ComplexNum retOutput(0, 0);
	retOutput.real(realOut);
	retOutput.imag(imgOut);

	return retOutput;
}
