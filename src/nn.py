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
 
# class NeuralNetwork_t:
#   def __init__ (self, topology):
#     self.num_layers = len (topology);
    
#     self.layers = [];
#     self.layer_weights = [];
#     self.layer_biases = [];
    
#     # Initialize layers 
#     for i in range (self.num_layers):
#       self.layers.append (Matrix (topology[i], 1));
    
#     # Initialize Layer Weight Matrices
#     for i in range (self.num_layers - 1):
#       self.layer_weights.append (Matrix (topology[i], topology [i + 1]));
  
#   def set_input_layer (self, input):
#     self.layers[0].matrix_from_container (input);
  
#   # Takes in a batch to propogate through network and 
#   # returns the result vector (final output activation values)
#   def forward (self, batch) -> Matrix:
#     i = 1;
#     res = Matrix (self.num_layers, 1);
#     while (i < self.num_layers):
#       res[i][0] = relu (self.layer_weights[i-1] * self.layers[i-1]);
#       i += 1;
#     return res;
#   # Computes the loss between actual matrix and expected matrix
#   def compute_cost_gradient(self, expected) -> int:
  
#   def backpropogation (cost_gradient):
 
#   # Performs a Rectified Linear Unit on matrix M  
#   def relu (self, m):
#     return m;

#   def __str__ (self):
#     res = "";
#     for i in range (self.num_layers):
#       res += self.layers[i].__str__() + "\n";
#     return res;

#============================================



# Represents a single layer in the neural network.
# Maintains the layer weights matrix and bias matrix
class Layer:
   def __init__(self, numNeurons, numNeuronsPrev):
      self.rows = numNeurons;
      self.cols = numNeuronsPrev;
      self.weights = np.zeros((numNeurons, numNeuronsPrev));
      self.bias = np.zeros(numNeurons);
    

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
    assert 0 <= layerIdx <= self.count
    return self.__layers[layerIdx]

class NeuralNetwork:
   def __init__ (self, topology):
      self.layers = Layers(topology);
   
   def __str__(self):
    str_repr = ""
    for layerIdx in range(self.layers.count):
        layer: Layer = self.layers[layerIdx]
        str_repr += "Layer " + str(layerIdx) + " - Weights: "
        str_repr += "#Rows=" + str(layer.weights.shape[0]) + ", "
        str_repr += "#Cols=" + str(layer.weights.shape[1]) + "\n"
        str_repr += str(layer.weights)
        str_repr += "\n\n"
    return str_repr
      

#The following is a mini-batch stochastic descent - i.e one sample at a time 
#instead of whole batches of samples

iterations = 10;
# dataset = none; // Todo 

for step in range (iterations):
  
  #Draw random sample from the dataset
  sample_index = getRandomSample(dataset);
  
  # Propogate sample input through network
  out_pred = forwardPropagate(nn, dataset[sample_index].sample);

  # Calculate Loss Gradient Tensor
  gradient = backwardPropagate(nn, out_pred, dataset[sample_index].actual_label);

  # Update neural network parameters using the gradient
  optimize(nn, gradient);

####
def optimize(nn, gradient):
   # Iterate Through each layer and update the weight matrix
   int i = 0;
   for each layerMatrix in nn.layers:
      layerMatrix += gradient

def getRandomSample(dataset):
  pass;

# ################################################################################
#                      """Forward Propagation Functions"""
# ################################################################################
# # Propagate input sample SAMPLE, through neural network 'NN'
# def forwardPropagate(NeuralNetwork nn, Dataset sample) -> array:
#   # Intilize output tensor, and set it's first layer activation vector to sample
#   out = np.zeros((nn.layers.count), 1);
#   out[0][0] = sample;

#   # Iteratively Propagate 
#   for layerIdx in range(0, nn.layers.count):
#     layer = nn.layer[layerIdx];
#     out[layerIdx] = propagateThroughLayer(layer.weights, layer.bias, out, layerIdx);
  
#   return out

# # Propagates Through a single layer -> Returns a Vector
# def propagateThroughLayer(weights, bias, activations, layerIdx):
#   z = linearTransform (weights, bias, activations, layerIdx)
#   finalizedActivation = activationFunction(z)
#   return finalizedActivation;

# def linearTransform(weights, bias, activations, layerIdx):
#   return weights * activations[layerIdx - 1] + bias;

# def activationFunction(tensor):
#   # Do relu?
#   pass

# ################################################################################
#                 """Backward Propagation Functions"""
# ################################################################################

# backwardPropagate(nn, activations, predictions):
#   gradient = np.array(nn.layers.count, );
#   # Calculate Initial Last Layer gradients
#   L = -1;
#   for k in range (len(nn.layers[-1])):
#    gradient[L][k] = dloss(activations[L], predictions) * dActivation(act[L]) * dLinearTransform (activations, L);
#   pass;
  
#   # Recursively Calculate gradient for the rest of the hidden layers
#   for layerIdx in rasnge(nn.layers.count - 1, 0, -1):
   
#    # For each node 'k' in this layer
#    for k in range(nn.layers[layerIdx].rows):
#     # Calculate δC/δwᴸₖ = sumOf(δaˡₖ/δzˡₖ * δzˡₖ/δwˡₖ * gradient[L + 1][j]) ∀j /
#     for j in range(nn.layerslayerIdx + 1):
#       gradient[layerIdx][k] += dActivation(act[layerIdx][k]) * \
#                                dTransform(act[layerIdx][k])  * \
#                                gradient[layerIdx + 1][j];
  

# def dLoss(np.array activations, prediction):
#   if (activation.shape != prediction.shape):
#     PANIC ("Shape of activation tensor is not equal to shape of prediction tensor");
  
#   return 
#   pass

# def dActivation(activation):
#   pass
# # Derivative of Linear Transform
# def dLinearTransform ()
