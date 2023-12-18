import numpy as np
from nn import *
from util import *

# Create a neural network
nn = NeuralNetwork([2, 10, 1, 1])
iterations = 1000

log("************* Initialized Network - Printing Now *************", bcolors.HEADER)
logc(str(nn))

for step in range(iterations):
    log("************* Starting Step: " + str(step) + " *************", bcolors.OKBLUE)

    # Draw random sample from the dataset
    sample = getRandomSample()
    log("Chosen Sample=" + str(sample), bcolors.OKCYAN)

    # Propagate sample input through network
    log("==== Starting Forward Propagation ====", bcolors.WARNING)
    outPred = forwardPropagate(nn, sample)
    showForwardProp(outPred)
    log("==== Finished Forward Propagation ====", bcolors.WARNING)
    
    # Calculate Loss Gradient Tensor
    actualLabel = getSampleLabel(sample)
    log("actualLabel:" + str(np.array(actualLabel)), bcolors.OKGREEN)
    
    log("==== Starting Backward Propagation ====", bcolors.WARNING)
    gradient = backwardPropagate(nn, outPred, np.array([actualLabel]))
    showBackwardProp(gradient)
    log("==== Finished Backward Propagation ====", bcolors.WARNING)

    # Update neural network parameters using the gradient
    log("==== Optimizing Network Weights With Gradient ====", bcolors.OKBLUE)
    optimize(nn, gradient, .10)
    log("==== Finished Optimization ====", bcolors.OKBLUE)
    log("==== Network Layers After Optimization", bcolors.HEADER)
    logc(str(nn))

def evaluateNetwork(nn: NeuralNetwork, numTests: int):
    MSE = 0
    for i in range(numTests):
        sample = getRandomSample()
        actualLabel = getSampleLabel(sample)
        outPred = forwardPropagate(nn, sample)
        
        predictedValue = outPred[-1] 
        # MSE += np.absolute(predictedValue - actualLabel)
        MSE += (predictedValue - actualLabel) ** 2

    MSE /= numTests
    print(f"MSE: {MSE}")

evaluateNetwork(nn, 10)
