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
      self.bias = np.random.rand(numNeurons, 1);
      print ("self.bias.shape", self.bias.shape);
    

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
# Propagate input sample SAMPLE, through neural network 'NN'
def forwardPropagate(nn, sample):
    activations = []  # Initialize with input sample
    activations.append(np.array(sample));
    for layer in nn.layers:
        
        z = linearTransform(layer.weights, layer.bias, activations[-1])
        a = activationFunction(z)
        print ("nn.fwp:a=", str(z));
        activations.append(np.array(a))
    assert (len(activations) == nn.layers.count + 1);
    print("--Printing Activations--")
    for actVec in activations:
        print(str(actVec))
    print("----Done----")

    return activations

def linearTransform(weights, bias, activation):
    z = np.dot(weights, activation) + bias
    return z

def relu(activation):
  result = np.array(activation)
  for i in range(len(activation)):
    if activation[i] > 0:
      result[i] = activation[i]
    else:
      result[i] = .33 * activation[i];
  return result

def activationFunction(tensor):
    # return relu(tensor);
    return np.where(tensor > 0, tensor, .01 * tensor) # activations[layerIdx])
    # return np.maximum(, tensor)  # ReLU activation function

# ################################################################################
#                 """Backward Propagation Functions"""
# ################################################################################
def backwardPropagate(nn, activations, actual) -> Layers:  
  gradient = Layers(nn.topology);
  # Calculate Initial Last Layer gradients
  layerIdx = -1
  delta = np.multiply(dActivation(activations, layerIdx), dLoss(activations, actual, layerIdx))
  gradient[layerIdx].weights = np.dot(delta, activations[layerIdx-1].T)  
  # gradient[layerIdx].weights = np.dot(delta.reshape(1, -1), activations[layerIdx-1].reshape(-1, 1).T)
  gradient[layerIdx].bias = delta;
  print("nnpy.bias", gradient[layerIdx].bias.shape);
  # wlTrans = np.transpose(gradient[layerIdx].weights)
  # fPrimeZ = dActivation(activations, layerIdx)
  # dotProduct = np.dot(wlTrans, delta)
  # delta = np.multiply(dotProduct.reshape(fPrimeZ.shape), fPrimeZ)

  wlTrans = np.transpose(gradient[layerIdx].weights);
  fPrimeZ = dActivation(activations, layerIdx)
  delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ);

  layerIdx -= 1;
  print("len(activations)", len(activations));
  while (layerIdx > -len(activations)): #-1 * (nn.layers.count)):
    # print ("cock");
    # δC/δwˡ = (δC/δaˡ⁻¹) * aˡ :: watch video for this part
   #gradient[layerIdx].weights = np.dot(delta.reshape(1, -1), activations[layerIdx - 1].reshape(-1, 1).T)
   
    gradient[layerIdx].weights = np.dot(delta, activations[layerIdx - 1].T)
    # gradient[layerIdx].weights = np.dot(delta.reshape(-1, 1), activations[layerIdx - 1].reshape(1, -1))
    gradient[layerIdx].bias = delta;
    
    # print("\nnnpy.bias", gradient[layerIdx].bias.shape);
    # gradient[layerIdx].weights = np.dot(delta, np.transpose(activations[layerIdx - 1]))
    # wlTrans = gradient[layerIdx].weights.T
    # fPrimeZ = dActivation(activations, layerIdx)
    # Ensure delta is a column vector
    # delta = delta.reshape(-1, 1)
    # Calculate dot product and ensure it's compatible with fPrimeZ shape
    # dotProduct = np.dot(wlTrans, delta)
    # dotProduct = np.dot(wlTrans, delta).reshape(fPrimeZ.shape)
    # delta = np.multiply(dotProduct, fPrimeZ)

    wlTrans = np.transpose(gradient[layerIdx].weights);
    fPrimeZ = dActivation(activations, layerIdx)
    delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ);

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

def drelu(activation):
  result = np.array(activation)
  for i in range(len(activation)):
     result[i] = 1.0 if activation[i] > 0 else 0.33
  
  return result

#Derivative of the activation function
def dActivation(activations, layerIdx):
    # return drelu(activations[layerIdx]);
    return np.where(activations[layerIdx] > 0, 1, .01) # activations[layerIdx])


# def dActivation(activations, layerIdx):
#   print ("DACTIVATION=" + str(np.ceil(activationFunction(activations[layerIdx]))))
#   return np.ceil(activationFunction(activations[layerIdx]));
  
# Derivative of Linear Transform
def dLinearTransform (activations, layerIdx):
  print ("DLINEARTRANS=" + str (activations[layerIdx - 1]))
  return activations[layerIdx - 1];

def optimize (nn : NeuralNetwork, gradient : Layers, step):
  for layerIdx in range(nn.layers.count):
    print ("shape=", gradient[layerIdx].weights.shape);
    print ("shape=", nn.layers[layerIdx].weights.shape);
    print ("bias.shape=", nn.layers[layerIdx].bias.shape);
    print ("gradient.shape=", gradient[layerIdx].bias.shape)
    nn.layers[layerIdx].weights -= gradient[layerIdx].weights * step;
    nn.layers[layerIdx].bias -= gradient[layerIdx].bias * step;

def lossFunction (output: np.array, actual : np.array):
  if not np.array_equal(output, actual):
    return 1;
  else:
    return 0;
    
  #return output - actual    