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

nn = NeuralNetwork([2, 1])
iterations = 1000
print(bcolors.HEADER + "*************Initialized Network - Printing Now*************" + bcolors.ENDC)
print(nn)

for step in range(iterations):  
  print(bcolors.OKBLUE + "*************Starting Step: " + str(step) + "*************" + bcolors.ENDC)
  
  # Draw random sample from the dataset
  # ssample = (.25, .25) #
  sample = getRandomSample()
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

def calcAcc (expected, actual):
  return 100 - ((expected - actual / expected) * 100)

def evaluateNetwork(nn: NeuralNetwork, numTests: int):
    MSE = 0
    for i in range(numTests):
        sample = getRandomSample()
        actualLabel = sum(sample)
        outPred = forwardPropagate(nn, sample)
        
        # If it's a regression problem:
        predictedValue = outPred[-1]  # Assuming outPred[-1] is the final predicted value
        MSE += (predictedValue - actualLabel)**2

        # If it's a classification problem and you need accuracy:
        # predictedLabel = np.argmax(outPred[-1]) 
        # numCorrect += (predictedLabel == actualLabel)

    MSE /= numTests
    print(f"MSE: {MSE}")

    # If calculating accuracy:
    # accuracy = numCorrect / numTests
    # print(f"Accuracy: {accuracy}")


evaluateNetwork(nn, 10);