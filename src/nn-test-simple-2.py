import numpy as np
from nn import *
from util import *

# Create a neural network
nn = NeuralNetwork([2, 10, 1, 1])
iterations = 1000

for step in range(iterations):
    # Draw random sample from the dataset
    sample = getRandomSample()
    
    # Propagate sample input through network
    outPred = forwardPropagate(nn, sample)
  
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
        
        predictedValue = outPred[-1] 
        MSE += np.absolute(predictedValue - actualLabel)
        # MSE += (predictedValue - actualLabel) ** 2

    MSE /= numTests
    print(f"MSE: {MSE}")

evaluateNetwork(nn, 10)
