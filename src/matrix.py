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
    def __multiply__ (self, other):
        res = Matrix (self.row, other.col);
        
        for r in range (self.row):
            for c in range (other.col):
                entry = 0;
                for ra in range (self.col):
                    entry += self.con[r][ra] * other.con[ra][c];
                res[r][c] = entry;
        
        return res;

    def __mul__ (self, other):
        if (self.col != other.row or self.row != other.col):
            PANIC ("Tensor Multiplication Collision");
        else:
            return self.__multiply__ (other);    
            
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
    
# Test
rows = 10
cols = 10
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
mat3 = mat * mat2;
print (mat3);


