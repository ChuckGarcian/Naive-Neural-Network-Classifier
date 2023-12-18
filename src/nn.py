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
    return activations

def linearTransform(weights, bias, activation):
    z = np.dot(weights, activation) + bias
    return z

def leakyRelu(tensor):
  return np.where(tensor > 0, tensor, .01 * tensor) # activations[layerIdx])

def relu(tensor):
  return np.maximum(0, tensor)  # ReLU activation function

def activationFunction(tensor):
    return leakyRelu(tensor);
    
# ################################################################################
#                 """Backward Propagation Functions"""
# ################################################################################
def backwardPropagate(nn, activations, actual) -> Layers:
  gradient = Layers(nn.topology)
  # Calculate Initial Last Layer gradients
  layerIdx = -1
  zL = linearTransform(nn.layers[layerIdx].weights, nn.layers[layerIdx].bias, activations[layerIdx-1])
  zL = arrToVec(zL)
  delta = np.multiply(dActivation(zL), dLoss(activations, actual, layerIdx))
  gradient[layerIdx].weights = np.dot(delta, activations[layerIdx - 1].T)
  gradient[layerIdx].bias = delta
  assert (gradient[layerIdx].weights.shape == nn.layers[layerIdx].weights.shape)
  assert (gradient[layerIdx].bias.shape == nn.layers[layerIdx].bias.shape)

  # Calculate Rest Layer Gradients using for loop
  for layerIdx in reversed(range(-nn.layers.count, -1)):
      zL = linearTransform(nn.layers[layerIdx].weights, nn.layers[layerIdx].bias, activations[layerIdx - 1])
      zL = arrToVec(zL)
      wlTrans = np.transpose(nn.layers[layerIdx + 1].weights)
      fPrimeZ = dActivation(zL)

      delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ)
      gradient[layerIdx].weights = np.dot(delta, activations[layerIdx - 1].T)
      gradient[layerIdx].bias = delta

      assert (gradient[layerIdx].weights.shape == nn.layers[layerIdx].weights.shape)
      assert (gradient[layerIdx].bias.shape == nn.layers[layerIdx].bias.shape)
  return gradient;

#Derivative of the loss function
def d_mae(y_pred, y_true):
    return np.where(y_pred > y_true, 1, -1)

def d_mse(y_pred, y_true):
  return 2 * (y_pred -  y_true)

def dLoss(activations, y_actual, layerIdx):
  # return d_mse(activations[layerIdx], y_actual);
  return d_mae(activations[layerIdx], y_actual);

def dRelu(zL):
  return np.where(zL > 0, 1, 0)

def dLeakyRelu(zL):     
  return np.where(zL > 0, 1, .01)

#Derivative of the activation function
def dActivation(zL):
    #return dRelu(zL)
    return  dLeakyRelu(zL)

def optimize (nn : NeuralNetwork, gradient : Layers, step):
  for layerIdx in range(nn.layers.count):
    nn.layers[layerIdx].weights -=  step * gradient[layerIdx].weights
    nn.layers[layerIdx].bias -= step * gradient[layerIdx].bias;
