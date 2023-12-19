import numpy as np
from nn import *
from util import *
from nn_functions import *

# Create a neural network
nn = NeuralNetwork([1, 1], Activations.LEAKY_RELU, LossFuncs.MSE)
iterations = 100

for step in range(iterations):
    # Draw random sample from the dataset
    sample = getRandomSample()

    # Propagate sample input through network
    outPred = forwardPropagate(nn, sample)
    # showForwardProp(outPred)

    # Calculate Loss Gradient Tensor
    actualLabel = getSampleLabel(sample)
    gradient = backwardPropagate(nn, outPred, np.array([actualLabel]))

    # Update neural network parameters using the gradient
    optimize(nn, gradient, .01)


def evaluateNetwork(nn: NeuralNetwork, numTests: int):
    MSE = 0
    for i in range(numTests):
        sample = getRandomSample()
        actualLabel = getSampleLabel(sample)
        outPred = forwardPropagate(nn, sample)
        showForwardProp(outPred)

        predictedValue = outPred[-1]
        # MSE += np.absolute(predictedValue - actualLabel)
        MSE += (predictedValue - actualLabel) ** 2

    MSE /= numTests
    print(f"MSE: {MSE}")


evaluateNetwork(nn, 10)
