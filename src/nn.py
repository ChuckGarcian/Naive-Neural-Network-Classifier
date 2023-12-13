from matrix import Matrix

"""
Neural Network 
"""

"""
Sketching:  
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
  
  # def set_input_layer ():
  def forward (self):
    i = 1;
    while (i < self.num_layers):
      self.layers[i] = self.layer_weights[i-1] * self.layers[i-1];
      print (self.layers[i]);
      i += 1;
    
      
  
  def __str__ (self):
    res = "";
    for i in range (self.num_layers):
      res += self.layers[i].__str__() + "\n";
    return res;

nn = NeuralNetwork ([10,10,10]);
nn.forward();