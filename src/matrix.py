"""
Basic Matrix Class - Key Features:
  * Construct that takes in initial number rows and collumns
  * Naive matrix add function - Takes two matrices and returns a new third matrix
    containing their sum
  * Naive Matrix Mult - Takes two matrices and returns a new third matrix containing
    the dot product of the two matrices

"""
def PANIC (msg):
    raise Exception (msg);

class Matrix:
    def __init__(self, row, col):
        self.con = []
        self.col =  col;
        self.row = row;
        for r in range(row):
            entry = []
            for c in range(col):
                entry.append(0)
            self.con.append(entry)
    def matrix_from_container (self, data):
        self.con = data;
        self.row = len (data);
        self.col = len (data[0]);
    
    ## Access Functions ##
    def __setitem__ (self, k, val):
        self.con[k] = val;
    
    def __getitem__ (self, k):
        return self.con[k]
    
    ## Matrix Operation Functions ##
    def __mul__ (self, other):
        if (self.col != other.row or self.row != other.col):
            PANIC ("Tensor Multiplication Collision");
        else:
            return __multiply__ (self, other);    
            
    def __add__ (self, other):
        if (self.col != other.col or self.row != other.row):
            PANIC ("Tensor Addition Collision - Attempted to add two different sized tensors ");
        else:
            res = Matrix (self.row, self.col);
            for r in range (self.row):
                for c in range (self.col):
                    res[r][c] = self[r][c] + other[r][c];
            
            return res;

    def ones (self):
        for r in range (self.row):
            for c in range (self.col):
                self.con[r][c] = 1;
                
    def __str__(self):
        s = ''
        # for r in self.data:
        #     s += str(r) + '\n'
        
        for r in range (self.row):
            s += str(self.con[r]) + '\n';
        return s
 
def __multiply__ (a, b):
        res = Matrix(a.row, b.col)
        for r in range(a.row):
            for c in range(b.col):
                entry = 0
                for ra in range(a.col):
                    entry += a.con[r][ra] * b.con[ra][c]
                res.con[r][c] = entry  # Correctly accessing the result matrix
        return res

# Test
rows = 5
cols = 5
mat = Matrix(rows, cols)

for r in range(rows):
    for c in range(cols):
        mat.con[r][c] = c

r = 0;
c = 1;
print ("R=", r);
print ("C=", c);
print ("Entry mat=", mat[r][c]);
print (mat);

mat2 = Matrix (rows, cols);
mat2.ones();
mat3 = mat2 * mat;
print (mat3);


