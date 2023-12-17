#from matrix import Matrix
import numpy as np

"""
Neural Network 
"""

"""
Sketching:
  -- Initialization -- 
  1. Intialize the neural network with random hidden layer weights
  -- Forward Propagation --
  1. Take in a sample input layer vector: x
  Initial Base Case:
  2. Calculate the activation values in the first hidden layer by performing a 
     linear transformation using the weights W¹ ⟹ a¹ = w¹x + b¹
  General Case:
  3. For every i'th layer, such that i is strictly a hidden layer or output layer (not an input
     layer), set the i'th layer's activation vector aⁱ = wⁱ*aⁱ⁻¹ + bⁱ.

  -- Computing cost gradient vector --
  loss_function - C = (aᴸ - y)²
  result of linear transformation - Z = wx + b
  activation_function - a = relu (z)
  
  ⟹
  C(w) = a(z(w)) ⟹ grad(C, W) = grad(c, a) * grad(a, z) * grad(z, w)
  
  ⟹
  grad(c,a) = 2(aᴸ - y) 
  grad(a, z) = δ(relu(z)) = w?
  grad(z, w) = x = aᴸ⁻¹

  In other words:
  1.  For every i'th layer starting from i
  -- Performing Layer Weight Optimization using the gradient Vector --

"""
# Represents a single layer in the neural network.
# Maintains the layer weights matrix and bias matrix
class Layer:
   def __init__(self, numNeurons, numNeuronsPrev):
      self.rows = numNeurons;
      self.cols = numNeuronsPrev;
      self.weights = np.random.rand(numNeurons, numNeuronsPrev) 
      self.bias = np.random.rand(numNeurons);
    

# Represents a fully connected neural network.
# Topology is a tuple that specifies the number of neurons in each layer.
class Layers:
  def __init__(self, topology):
    self.count = len(topology) - 1
    self.__layers = [None] * self.count

    # Create 'count' many layer objects
    for layerIdx in range(self.count):
      numNeuronsPrev = topology[layerIdx]
      numNeurons = topology[layerIdx + 1]
      self.__layers[layerIdx] = Layer(numNeurons, numNeuronsPrev)

  def __getitem__(self, layerIdx):
    return self.__layers[layerIdx]
  
  def __str__(self):
    str_repr = ""
    for layerIdx in range(self.count):
        layer: Layer = self.__layers[layerIdx]
        str_repr += "Layer " + str(layerIdx) + "\n"
        str_repr += "Weights - #Rows=" + str(layer.weights.shape[0]) + ", "
        str_repr += "#Cols=" + str(layer.weights.shape[1]) + ":\n"
        str_repr += str(layer.weights)
        # str_repr += "\n" + self.getBiasStr(layer);
        
    return str_repr

class NeuralNetwork:
   def __init__ (self, topology):
      self.topology = topology;
      self.layers = Layers(topology);
   
   def getBiasStr(self, layer : Layer):
    str_repr = "Bias - Entries=" + str(len(layer.bias)) + ":\n"
    for b in layer.bias:
      str_repr += str(b)
      str_repr += "\n"
    return str_repr

   def __str__(self):
    str_repr = ""
    for layerIdx in range(self.layers.count):
        layer: Layer = self.layers[layerIdx]
        str_repr += "Layer " + str(layerIdx) + "\n"
        str_repr += "Weights - #Rows=" + str(layer.weights.shape[0]) + ", "
        str_repr += "#Cols=" + str(layer.weights.shape[1]) + "\n"
        str_repr += str(layer.weights)
        str_repr += "\n" + self.getBiasStr(layer);
      
    return str_repr


#The following is a mini-batch stochastic descent - i.e one sample at a time 
#instead of whole batches of samples


# dataset = none; // Todo 

# ####
# def optimize(nn, gradient):
#    # Iterate Through each layer and update the weight matrix
#    int i = 0;
#    for each layerMatrix in nn.layers:
#       layerMatrix += gradient


# ################################################################################
#                      """Forward Propagation Functions"""
# ################################################################################

#Propagate input sample SAMPLE, through neural network 'NN'
def forwardPropagate(nn : NeuralNetwork, sample : tuple) -> list:
  # Initialize output tensor, and set it's first layer activation vector to sample
  activations = [];
  activations.append(np.array(sample));
  
  # Iteratively Propagate 
  for layerIdx in range(0, nn.layers.count):
    layer = nn.layers[layerIdx];
    activations.append(propagateThroughLayer(layer.weights, layer.bias, activations, layerIdx + 1));
  
  print ("--Printing Activations--")
  for actVec in activations:
    print (str(actVec))
  print ("----Done----");
    
  return activations

# Propagates Through a single layer -> Returns a Vector
def propagateThroughLayer(weights, bias, activations, layerIdx) -> np.array:
  z = linearTransform (weights, bias, activations, layerIdx)
  finalizedActivation = activationFunction(z)
  return finalizedActivation;

def linearTransform(weights, bias, activations, layerIdx):
  mat = np.dot (weights, activations[layerIdx - 1]);
  res = mat + np.transpose(bias);
  return  res;

def activationFunction(tensor):
  return np.maximum(0, tensor)
  

# ################################################################################
#                 """Backward Propagation Functions"""
# ################################################################################

def backwardPropagate(nn, activations, actual) -> Layers:  
  gradient = Layers(nn.topology);
  # Calculate Initial Last Layer gradients
  layerIdx = -1
  delta = np.multiply(dActivation(activations, layerIdx), dLoss(activations, actual, layerIdx))
  gradient[layerIdx].weights = np.dot(delta.reshape(1, -1), activations[layerIdx-1].reshape(-1, 1).T)

  # wlTrans = np.transpose(gradient[layerIdx].weights)
  # fPrimeZ = dActivation(activations, layerIdx)
  # dotProduct = np.dot(wlTrans, delta)
  # delta = np.multiply(dotProduct.reshape(fPrimeZ.shape), fPrimeZ)

  wlTrans = np.transpose(gradient[layerIdx].weights);
  fPrimeZ = dActivation(activations, layerIdx)
  delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ);

  layerIdx -= 1;
  while (layerIdx >= -1 * (nn.layers.count)):
    # δC/δwˡ = (δC/δaˡ⁻¹) * aˡ :: watch video for this part
   #gradient[layerIdx].weights = np.dot(delta.reshape(1, -1), activations[layerIdx - 1].reshape(-1, 1).T)
    gradient[layerIdx].weights = np.dot(delta.reshape(-1, 1), activations[layerIdx - 1].reshape(1, -1))

    # gradient[layerIdx].weights = np.dot(delta, np.transpose(activations[layerIdx - 1]))
    wlTrans = gradient[layerIdx].weights.T
    fPrimeZ = dActivation(activations, layerIdx - 1)
    # Ensure delta is a column vector
    delta = delta.reshape(-1, 1)
    # Calculate dot product and ensure it's compatible with fPrimeZ shape
    dotProduct = np.dot(wlTrans, delta).reshape(fPrimeZ.shape)
    delta = np.multiply(dotProduct, fPrimeZ)
    #Compute error delta for layerIdx
    # δˡ = ((Wˡ)ᵀ * δˡ⁺¹ ) ⊙ f′(zˡ)
    # wlTrans = np.transpose(gradient[layerIdx].weights);
    # fPrimeZ = dActivation(activations, layerIdx)
    # delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ);
    layerIdx -= 1
  print ("---Computed Gradient---")
  print ("---Printing Gradient Now")
  print (str(gradient));
  print ("---Finished Printing---");
  
  return gradient;

#Derivative of the loss function
def dLoss(activations, prediction, layerIdx):
  print ("DLOS.ACTUAL=" + str(activations[layerIdx]));
  print ("DLOS.PREDICTION=" + str(prediction));
  print ("DLOS.CALC=" + str(2 * (activations[layerIdx] - prediction)))
  return 2 * (activations[layerIdx] - prediction);

#Derivative of the activation function
def dActivation(activations, layerIdx):
    return np.where(activations[layerIdx] > 0, 1, 0)

# def dActivation(activations, layerIdx):
#   print ("DACTIVATION=" + str(np.ceil(activationFunction(activations[layerIdx]))))
#   return np.ceil(activationFunction(activations[layerIdx]));
  
# Derivative of Linear Transform
def dLinearTransform (activations, layerIdx):
  print ("DLINEARTRANS=" + str (activations[layerIdx - 1]))
  return activations[layerIdx - 1];

def optimize (nn : NeuralNetwork, gradient : Layers):
  for layerIdx in range(nn.layers.count):
    nn.layers[layerIdx].weights -= 0.1 * gradient[layerIdx].weights;
  

def lossFunction (output: np.array, actual : np.array):
  if not np.array_equal(output, actual):
    return 1;
  else:
    return 0;
    
  #return output - actual    