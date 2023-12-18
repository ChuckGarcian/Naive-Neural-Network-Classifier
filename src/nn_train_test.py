import numpy as np
from nn import *
from util import *

# Create a neural network
nn = NeuralNetwork([2, 10,  1])
iterations = 1000

log("************* Initialized Network - Printing Now *************", bcolors.HEADER)
print(nn)

for step in range(iterations):
    log("************* Starting Step: " + str(step) + " *************", bcolors.OKBLUE)

    # Draw random sample from the dataset
    sample = getRandomSample()
    log("Chosen Sample=" + str(sample), bcolors.OKCYAN)

    # Propagate sample input through network
    log("==== Starting Forward Propagation ====", bcolors.WARNING)
    outPred = forwardPropagate(nn, sample)
    log("==== Finished Forward Propagation ====", bcolors.WARNING)
    log("Output Prediction=" + str(outPred), bcolors.OKGREEN)

    # Calculate Loss Gradient Tensor
    actualLabel = sum(sample)

    log("actualLabel:" + str(np.array(actualLabel)), bcolors.OKGREEN)
    loss = lossFunction(np.array(outPred[-1]), np.array(actualLabel))
    log("Loss=" + str(loss), bcolors.FAIL)
    log("==== Starting Backward Propagation ====", bcolors.WARNING)
    gradient = backwardPropagate(nn, outPred, np.array([actualLabel]))
    log("==== Finished Backward Propagation ====", bcolors.WARNING)

    # Update neural network parameters using the gradient
    log("==== Optimizing Network Weights With Gradient ====", bcolors.OKBLUE)
    optimize(nn, gradient, .01)
    log("==== Finished Optimization ====", bcolors.OKBLUE)
    log("==== Network Layers After Optimization", bcolors.HEADER)
    # log(str(nn))

def evaluateNetwork(nn: NeuralNetwork, numTests: int):
    MSE = 0
    for i in range(numTests):
        sample = getRandomSample()
        actualLabel = sum(sample)
        outPred = forwardPropagate(nn, sample)

        # If it's a regression problem:
        predictedValue = outPred[-1]  # Assuming outPred[-1] is the final predicted value
        MSE += np.absolute(predictedValue - actualLabel)

        # If it's a classification problem and you need accuracy:
        # predictedLabel = np.argmax(outPred[-1])
        # numCorrect += (predictedLabel == actualLabel)

    MSE /= numTests
    print(f"MSE: {MSE}")

    # If calculating accuracy:
    # accuracy = numCorrect / numTests
    # print(f"Accuracy: {accuracy}")


evaluateNetwork(nn, 10)
