from nn import *

# Returns a tuple of two random digits
def getRandomSample() -> tuple:
  return np.random.randint(0,100) /100;
  return (np.random.randint(0, 100) / 100, np.random.randint(0, 100) / 100);

nn = NeuralNetwork ([1,4, 1]);
iterations = 1;
print (nn);

for step in range (iterations):  
  #Draw random sample from the dataset
  sample = getRandomSample();
  
  # Propogate sample input through network
  outPred = forwardPropagate(nn, sample);
  print ("OutPred=", str(outPred));
  print ("Sample=", str(sample));
  # Calculate Loss Gradient Tensor
  actualLabel =  sample # np.sum(sample);
  loss = lossFunction(np.array(outPred[-1]), np.array(actualLabel));
  print ("Sum=" + str(actualLabel));
  print ("Loss=" + str(loss));
  gradient = backwardPropagate(nn, outPred, actualLabel);
  # Update neural network parameters using the gradient
  optimize(nn, gradient);
  print ("AfterBackProp Layers=" + str(nn));

  #Print State




