import numpy as np
from nn import *
from util import *
from nn_functions import *
from mnist_loader import *
from collections import deque

def getRandomSample(td) -> tuple:
    rt = np.random.randint(0, len(td))
    return td.pop(rt)

def getSampleLabel(sample):
    return sample[1]

def runningAccuracy(predictedValue, actualValue, results):
  results.append(int(predictedValue == actualValue))
  if (step % 1_000 == 0):
      accuracy = sum(results) / len(results)
      print(f"Accuracy: {accuracy * 100}%")  

def evaluateNetwork(nn: NeuralNetwork, testData: np.array, numTests: int):
    correct_predictions = 0
    tests_ran = 0
    for i in range(numTests):
        sample = getRandomSample(testData)
        actualLabel = getSampleLabel(sample)
        outPred = forwardPropagate(nn, sample[0])
      
        predictedValue = np.argmax(outPred[-1])

        if predictedValue == actualLabel:
            correct_predictions += 1
        tests_ran += 1
        if (i % 100 == 0):
            print((correct_predictions / tests_ran) * 100)

    accuracy = (correct_predictions / numTests) *  100  # Multiply by 100 to get percentage
    print(f"Accuracy: {accuracy}%")  # Add a '%' to the print statement

# Create a neural network
nn = NeuralNetwork([784, 64, 10], Activations.SIGMOID, LossFuncs.MSE)

data = load_data_wrapper()
training_data = list(data[0])
verification_data = list(data[1])
test_data = list(data[2])

iterations = len(training_data)
results = deque(maxlen=200)

for step in range(iterations):
  # Draw random sample from the dataset
  sample = getRandomSample(training_data)

  # Propagate sample input through network
  outPred = forwardPropagate(nn, sample[0])
    
  # Calculate Loss Gradient Tensor
  actualLabel = getSampleLabel(sample)
  gradient = backwardPropagate(nn, outPred, actualLabel)

  # Update neural network parameters using the gradient
  optimize(nn, gradient, .1)
  runningAccuracy(np.argmax(outPred[-1]), np.argmax(actualLabel), results)
  
print("Done Training Network")
evaluateNetwork(nn, test_data, len(test_data))
