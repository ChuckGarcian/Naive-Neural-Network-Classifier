from nn import *

# Returns a tuple of two random digits
def getRandomSample() -> tuple:
  return (np.random.randint(0, 100) / 100, np.random.randint(0, 100) / 100);

nn = NeuralNetwork ([2, 10, 10 ,1]);
iterations = 10;
print (nn);

for step in range (iterations):  
  #Draw random sample from the dataset
  sample = getRandomSample();
  
  # Propogate sample input through network
  out_pred = forwardPropagate(nn, sample);
  
  # Calculate Loss Gradient Tensor
  # actual_label = np.sump(sample, axis=1);
  # gradient = backwardPropagate(nn, out_pred, actual_label);

  # Update neural network parameters using the gradient
  # optimize(nn, gradient);

  #Print State




