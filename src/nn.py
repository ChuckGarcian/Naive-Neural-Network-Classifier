#from matrix import Matrix
import numpy as np
from util import *
import pdb

# Represents a single layer in the neural network.
# Maintains the layer weights matrix and bias matrix
class Layer:
   def __init__(self, numNeurons, numNeuronsPrev):
      self.rows = numNeurons;
      self.cols = numNeuronsPrev;
      self.weights = np.random.rand(numNeurons, numNeuronsPrev) 
      self.bias = np.random.rand(numNeurons, 1);
      logc ("self.bias.shape" + str(self.bias.shape));

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

# ################################################################################
#                      """Forward Propagation Functions"""
# ################################################################################

# Takes row vector (One Dimensional Array) and reshapes to column vector
def arrToVec(arr) -> np.array:
  return np.array(arr).reshape(-1, 1)


#Propagate input sample SAMPLE, through neural network 'NN'
def forwardPropagate(nn, sample):
    activations = []  # Initialize with input sample
    activations.append(arrToVec(sample))
    
    for layer in nn.layers:    
        z = linearTransform(layer.weights, layer.bias, activations[-1])
        a = activationFunction(z)
        logc ("nn.fwp:a=" + str(z));
        activations.append(a)
    
    assert (len(activations) == nn.layers.count + 1);
    
    logc("--logcing Activations--")
    for actVec in activations:
        logc(str(actVec))
        assert actVec.shape[1] == 1
    logc("----Done----")

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

def leakyRelu(tensor):
  return np.where(tensor > 0, tensor, .1 * tensor) # activations[layerIdx])

def relu(tensor):
  return np.maximum(0, tensor)  # ReLU activation function

def activationFunction(tensor):
    return leakyRelu(tensor);
    

# ################################################################################
#                 """Backward Propagation Functions"""
# ################################################################################
def backwardPropagate(nn, activations, actual) -> Layers:  
  gradient = Layers(nn.topology);
  # Calculate Initial Last Layer gradients
  layerIdx = -1
  
  zL = linearTransform(nn.layers[layerIdx].weights, nn.layers[layerIdx].bias, activations[layerIdx-1])
  zL = arrToVec(zL)
  delta = np.multiply(dActivation(zL), dLoss(activations, actual, layerIdx))
  log ("delta.shape=" + str(delta.shape), bcolors.FAIL);
  assert (delta.shape[0] == nn.layers[layerIdx].weights.shape[0])
  assert (activations[layerIdx - 1].shape[0] == nn.layers[layerIdx].weights.shape[1])
  
  gradient[layerIdx].weights = np.dot(delta, activations[layerIdx - 1 ].T)  
  gradient[layerIdx].bias = delta;

  # gradient[layerIdx].weights = np.dot(delta.reshape(1, -1), activations[layerIdx-1].reshape(-1, 1).T)
  log ("gradient.w.shape=" + str(gradient[layerIdx].weights.shape), bcolors.FAIL);
  log ("nn.w.shape="  + str(nn.layers[layerIdx].weights.shape),bcolors.FAIL);
  log ("nn.bias.shape=" + str( nn.layers[layerIdx].bias.shape), bcolors.FAIL);
  log ("gradient.bias.shape=" + str(gradient[layerIdx].bias.shape), bcolors.FAIL)
  
  assert (gradient[layerIdx].weights.shape == nn.layers[layerIdx].weights.shape)
  assert (gradient[layerIdx].bias.shape == nn.layers[layerIdx].bias.shape)
  logc("nnpy.bias" + str(gradient[layerIdx].bias.shape));
  # wlTrans = np.transpose(gradient[layerIdx].weights)
  # fPrimeZ = dActivation(activations, layerIdx)
  # dotProduct = np.dot(wlTrans, delta)
  # delta = np.multiply(dotProduct.reshape(fPrimeZ.shape), fPrimeZ)

  layerIdx -= 1
  logc("len(activations)" + str(len(activations)));
  while (layerIdx > -len(activations)): #-1 * (nn.layers.count)): 
    
    logc("LayerIdx=" + str(layerIdx))
    zL = linearTransform(nn.layers[layerIdx].weights, nn.layers[layerIdx].bias, activations[layerIdx - 1])
    zL = arrToVec(zL)  
    wlTrans = np.transpose(nn.layers[layerIdx + 1].weights);
    fPrimeZ = dActivation(zL)
    assert (activations[layerIdx - 1].shape[0] == nn.layers[layerIdx].weights.shape[1])
    
    delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ);
    log ("nn.layers[layerIdx].weights.shape[0])=" + str (nn.layers[layerIdx].weights.shape), bcolors.FAIL)
    log ("delta.shape=" + str(delta.shape), bcolors.FAIL);
    assert (delta.shape[0] == nn.layers[layerIdx].weights.shape[0])
    # δC/δwˡ = (δC/δaˡ⁻¹) * aˡ :: watch video for this part
   #gradient[layerIdx].weights = np.dot(delta.reshape(1, -1), activations[layerIdx - 1].reshape(-1, 1).T)
    
    # logc("LayerIdx=", layerIdx)
    # logc("layers.count", nn.layers.count + 1);
    
    if (-layerIdx == nn.layers.count + 1):
      break;
    gradient[layerIdx].weights = np.dot(delta, activations[layerIdx - 1].T)
    gradient[layerIdx].bias = delta;
    log ("gradient.w.shape=" + str(gradient[layerIdx].weights.shape), bcolors.FAIL);
    log ("nn.w.shape="  + str(nn.layers[layerIdx].weights.shape),bcolors.FAIL);
    log ("nn.bias.shape=" + str( nn.layers[layerIdx].bias.shape), bcolors.FAIL);
    log ("gradient.bias.shape=" + str(gradient[layerIdx].bias.shape), bcolors.FAIL)
    
    assert (gradient[layerIdx].weights.shape == nn.layers[layerIdx].weights.shape)
    assert (gradient[layerIdx].bias.shape == nn.layers[layerIdx].bias.shape)
    layerIdx -= 1
    # gradient[layerIdx].weights = np.dot(delta.reshape(-1, 1), activations[layerIdx - 1].reshape(1, -1))

    # logc("\nnnpy.bias", gradient[layerIdx].bias.shape);
    # gradient[layerIdx].weights = np.dot(delta, np.transpose(activations[layerIdx - 1]))
    # wlTrans = gradient[layerIdx].weights.T
    # fPrimeZ = dActivation(activations, layerIdx)
    # Ensure delta is a column vector
    # delta = delta.reshape(-1, 1)
    # Calculate dot product and ensure it's compatible with fPrimeZ shape
    # dotProduct = np.dot(wlTrans, delta)
    # dotProduct = np.dot(wlTrans, delta).reshape(fPrimeZ.shape)
    # delta = np.multiply(dotProduct, fPrimeZ)
    # zL = linearTransform(nn.layers[layerIdx].weights, nn.layers[layerIdx].bias, activations[layerIdx-1])
    # zL = arrToVec(zL)
    # wlTrans = np.transpose(gradient[layerIdx].weights);
    # fPrimeZ = arrToVec(dActivation(zL))
    # pdb.set_trace()
    # delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ);
    #Compute error delta for layerIdx
    # δˡ = ((Wˡ)ᵀ * δˡ⁺¹ ) ⊙ f′(zˡ)
    # wlTrans = np.transpose(gradient[layerIdx].weights);
    # fPrimeZ = dActivation(activations, layerIdx)
    # delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ);
    
  logc ("---Computed Gradient---")
  logc ("---logcing Gradient Now")
  logc (str(gradient));
  logc ("---Finished logcing---");
  
  return gradient;

#Derivative of the loss function
def dLoss(activations, prediction, layerIdx):
  logc ("DLOS.ACTUAL=" + str(activations[layerIdx]));
  logc ("DLOS.PREDICTION=" + str(prediction));
  logc ("DLOS.CALC=" + str(2 * (activations[layerIdx] - prediction)))
  return 2 * (activations[layerIdx] - prediction);

def drelu(activation):
  result = np.array(activation)
  for i in range(len(activation)):
     result[i] = 1.0 if activation[i] > 0 else 0.33
  
  return result

#Derivative of the activation function
def dActivation(zL):
    # return drelu(activations[layerIdx]);
    return np.where(zL > 0, 1, .1) # activations[layerIdx])


# def dActivation(activations, layerIdx):
#   logc ("DACTIVATION=" + str(np.ceil(activationFunction(activations[layerIdx]))))
#   return np.ceil(activationFunction(activations[layerIdx]));
  
# Derivative of Linear Transform
def dLinearTransform (activations, layerIdx):
  logc ("DLINEARTRANS=" + str (activations[layerIdx - 1]))
  return activations[layerIdx - 1];

def optimize (nn : NeuralNetwork, gradient : Layers, step):
  for layerIdx in range(nn.layers.count):
    nn.layers[layerIdx].weights -=  step * gradient[layerIdx].weights
    nn.layers[layerIdx].bias -= step * gradient[layerIdx].bias;

def lossFunction (output: np.array, actual : np.array):
  
  if not np.array_equal(output, actual):
    return 1;
  else:
    return 0;
    
  #return output - actual    