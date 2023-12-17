from nn import *

# Returns a tuple of two random digits
def getRandomSample() -> tuple:
  #return np.random.randint(0,100) /100;
  return (np.random.randint(0, 100) / 100, np.random.randint(0, 100) / 100);


# ANSI escape codes for colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

nn = NeuralNetwork([2, 3, 3, 1])
iterations = 100
print(bcolors.HEADER + "*************Initialized Network - Printing Now*************" + bcolors.ENDC)
print(nn)

for step in range(iterations):  
  print(bcolors.OKBLUE + "*************Starting Step: " + str(step) + "*************" + bcolors.ENDC)
  
  # Draw random sample from the dataset
  sample = (.25, .25) #getRandomSample()
  print(bcolors.OKCYAN + "Chosen Sample=" + str(sample) + bcolors.ENDC)
  
  # Propagate sample input through network
  print(bcolors.WARNING + "====Starting Forward Propagation====" + bcolors.ENDC)
  outPred = forwardPropagate(nn, sample)
  print(bcolors.WARNING + "====Finished Forward Propagation====" + bcolors.ENDC)
  print(bcolors.OKGREEN + "Output Prediction=" + str(outPred) + bcolors.ENDC)
  
  # Calculate Loss Gradient Tensor
  actualLabel = sum(sample)
  
  print ("actualLabel:" + str(np.array(actualLabel)));
  loss = lossFunction(np.array(outPred[-1]), np.array(actualLabel))
  print(bcolors.FAIL + "Loss=" + str(loss) + bcolors.ENDC)
  print(bcolors.WARNING + "====Starting Backward Propagation====" + bcolors.ENDC)
  gradient = backwardPropagate(nn, outPred, np.array([actualLabel]))
  print(bcolors.WARNING + "====Finished Backward Propagation====" + bcolors.ENDC)
  
  # Update neural network parameters using the gradient
  print(bcolors.OKBLUE + "====Optimizing Network Weights With Gradient====" + bcolors.ENDC)
  optimize(nn, gradient)
  print(bcolors.OKBLUE + "====Finished Optimization====" + bcolors.ENDC)
  print(bcolors.HEADER + "==== Network Layers After Optimization" + bcolors.ENDC)
  print(nn)

def evaluateNetwork(nn : NeuralNetwork, numTests : int):
  numCorrect = 0
  for i in range(numTests):
    sample = (.25, .25) #getRandomSample()
    actualLabel = sum(sample);
    outPred = forwardPropagate(nn, sample)
    predictedLabel = np.argmax(outPred[-1])  # Assuming outPred[-1] is a probability distribution
    if predictedLabel == actualLabel:
      numCorrect += 1
  accuracy = numCorrect / numTests
  print(f"Accuracy: {accuracy * 100}%")

evaluateNetwork(nn, 10);