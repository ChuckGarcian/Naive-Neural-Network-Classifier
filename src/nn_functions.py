import numpy as np

#Derivative of the loss function
def d_mae(y_pred, y_true):
    return np.where(y_pred > y_true, 1, -1)

def d_mse(y_pred, y_true):
  return 2 * (y_pred -  y_true)

def dRelu(zL):
  return np.where(zL > 0, 1, 0)

def dLeakyRelu(zL):     
  return np.where(zL > 0, 1, .01)

def leakyRelu(tensor):
  return np.where(tensor > 0, tensor, .01 * tensor) # activations[layerIdx])

def relu(tensor):
  return np.maximum(0, tensor)  # ReLU activation function

class Activations:
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"

class LossFuncs:
    MSE = "mse"
    MAE = "mae"

