from nn import *

# Returns a tuple of two random digits
def getRandomSample() -> tuple:
  return np.random.randint(0,100) /100;
  return (np.random.randint(0, 100) / 100, np.random.randint(0, 100) / 100);

nn = NeuralNetwork ([1, 1]);
iterations = 1;
print ("*************Initialized Network - Printing Now*************")
print (nn);

for step in range (iterations):  
  print ("*************Starting Step: " + str(step) + "*************");
  
  #Draw random sample from the dataset
  sample = getRandomSample();
  print ("Chosen Sample=", str(sample));
  
  # Propogate sample input through network
  print ("====Starting Forward Propagation====")
  outPred = forwardPropagate(nn, sample);
  print ("====Finished Forward Propagation====")
  print ("Output Prediction=", str(outPred));
  
  # Calculate Loss Gradient Tensor
  actualLabel =  sample # np.sum(sample);
  loss = lossFunction(np.array(outPred[-1]), np.array(actualLabel));
  print ("Loss=" + str(loss));
  print ("====Starting Backward Propagation====");
  gradient = backwardPropagate(nn, outPred, actualLabel);
  print ("====Finished Backward Propagation====")
  
  # Update neural network parameters using the gradient
  print ("====Optimizing Network Weights With Gradient====");
  optimize(nn, gradient);
  print ("====Finished Optimization====");
  print ("==== Network Layers After Optimization");
  print (nn);

  #Print State




