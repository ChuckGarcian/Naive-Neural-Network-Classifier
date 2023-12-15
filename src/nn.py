from matrix import Matrix

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
 
class NeuralNetwork:
  def __init__ (self, topology):
    self.num_layers = len (topology);
    
    self.layers = [];
    self.layer_weights = [];
    self.layer_biases = [];
    
    # Initialize layers 
    for i in range (self.num_layers):
      self.layers.append (Matrix (topology[i], 1));
    
    # Initialize Layer Weight Matrices
    for i in range (self.num_layers - 1):
      self.layer_weights.append (Matrix (topology[i], topology [i + 1]));
  
  def set_input_layer (self, input):
    self.layers[0].matrix_from_container (input);
  
  # Takes in a batch to propogate through network and 
  # returns the result vector (final output activation values)
  def forward (self, batch) -> Matrix:
    i = 1;
    res = Matrix (self.num_layers, 1);s
    while (i < self.num_layers):
      res[i][0] = relu (self.layer_weights[i-1] * self.layers[i-1]);
      i += 1;
    return res;
  # Computes the loss between actual matrix and expected matrix
  def compute_cost_gradient(self, expected) -> int:
  

  def backpropogation (cost_gradient):

  # Performs a Rectified Linear Unit on matrix M  
  def relu (self, m):
    return m;

  def __str__ (self):
    res = "";
    for i in range (self.num_layers):
      res += self.layers[i].__str__() + "\n";
    return res;

nn = NeuralNetwork ([10,10,10]);
nn.forward();