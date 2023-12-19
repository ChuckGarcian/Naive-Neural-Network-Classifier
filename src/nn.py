#from matrix import Matrix
import numpy as np
from util import *
import pdb
from nn_functions import *

#Derivative of the loss function
def d_mae(y_pred, y_true):
    return np.where(y_pred > y_true, 1, -1)

def d_mse(y_pred, y_true):
  return 2 * (y_pred -  y_true)

def relu(tensor):
  return np.maximum(0, tensor)  # ReLU activation function

def dRelu(zL):
  return np.where(zL > 0, 1, 0)

def leakyRelu(tensor):
  return np.where(tensor > 0, tensor, .1 * tensor) # activations[layerIdx])

def dLeakyRelu(zL):     
  return np.where(zL > 0, 1, .1)

def sigmoid(z):
    z = np.array(z)
    result = np.zeros_like(z)
    result[z >= 0] = 1. / (1. + np.exp(-z[z >= 0]))
    result[z < 0] = np.exp(z[z < 0]) / (1. + np.exp(z[z < 0]))
    return result

def dSigmoid(zL):
  return sigmoid(zL) * (1 - sigmoid(zL))

class Activations:
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SIGMOID = "sigmoid"

class LossFuncs:
    MSE = "mse"
    MAE = "mae"

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
   def __init__ (self, topology, activationFunc, lossFunc):
      self.topology = topology;
      self.layers = Layers(topology);
      self.__activationFunc = activationFunc
      self.__lossFunc = lossFunc
   
   def dLoss(self, activations, y_actual, layerIdx):
    if self.__lossFunc == LossFuncs.MSE:
     return d_mse(activations[layerIdx], y_actual)
    elif self.__lossFunc == LossFuncs.MAE:
     return d_mae(activations[layerIdx], y_actual)
   
   def dActivation(self, zL):
    if self.__activationFunc == Activations.RELU:
     return dRelu(zL)
    elif self.__activationFunc == Activations.LEAKY_RELU:
     return dLeakyRelu(zL)
    elif self.__activationFunc == Activations.SIGMOID:
      return sigmoid(zL)

   def activationFunction(self, tensor):
    if self.__activationFunc == Activations.RELU:
      return relu(tensor)
    elif self.__activationFunc == Activations.LEAKY_RELU:
      return leakyRelu(tensor)
    elif self.__activationFunc == Activations.SIGMOID:
      return sigmoid(tensor)

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

def arrToVec(arr) -> np.array:
  return np.array(arr).reshape(-1, 1)

def linearTransform(weights, bias, activation):
    z = np.dot(weights, activation) + bias
    return z

#Propagate input sample SAMPLE, through neural network 'NN'
def forwardPropagate(nn, sample):
    activations = []  # Initialize with input sample
    activations.append(arrToVec(sample))
    
    for layer in nn.layers:    
        z = linearTransform(layer.weights, layer.bias, activations[-1])
        a = nn.activationFunction(z)
        logc ("nn.fwp:a=" + str(z));
        activations.append(a)
    
    assert (len(activations) == nn.layers.count + 1);
    return activations

def backwardPropagate(nn, activations, actual) -> Layers:
  gradient = Layers(nn.topology)
  # Calculate Initial Last Layer gradients
  layerIdx = -1
  zL = linearTransform(nn.layers[layerIdx].weights, nn.layers[layerIdx].bias, activations[layerIdx-1])
  zL = arrToVec(zL)
  delta = np.multiply(nn.dActivation(zL), nn.dLoss(activations, actual, layerIdx))
  gradient[layerIdx].weights = np.dot(delta, activations[layerIdx - 1].T)
  gradient[layerIdx].bias = delta
  assert (gradient[layerIdx].weights.shape == nn.layers[layerIdx].weights.shape)
  assert (gradient[layerIdx].bias.shape == nn.layers[layerIdx].bias.shape)

  # Calculate Rest Layer Gradients using for loop
  for layerIdx in reversed(range(-nn.layers.count, -1)):
      zL = linearTransform(nn.layers[layerIdx].weights, nn.layers[layerIdx].bias, activations[layerIdx - 1])
      zL = arrToVec(zL)
      wlTrans = np.transpose(nn.layers[layerIdx + 1].weights)
      fPrimeZ = nn.dActivation(zL)

      delta = np.multiply(np.dot(wlTrans, delta), fPrimeZ)
      gradient[layerIdx].weights = np.dot(delta, activations[layerIdx - 1].T)
      gradient[layerIdx].bias = delta

      assert (gradient[layerIdx].weights.shape == nn.layers[layerIdx].weights.shape)
      assert (gradient[layerIdx].bias.shape == nn.layers[layerIdx].bias.shape)
  return gradient;

def optimize (nn : NeuralNetwork, gradient : Layers, step):
  for layerIdx in range(nn.layers.count):
    nn.layers[layerIdx].weights -=  step * gradient[layerIdx].weights
    nn.layers[layerIdx].bias -= step * gradient[layerIdx].bias;
