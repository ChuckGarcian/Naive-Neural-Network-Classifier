"""
Basic Matrix Class - Key Features:
  * Construct that takes in initial number rows and collumns
  * Naive matrix add function - Takes two matrices and returns a new third matrix
    containing their sum
  * Naive Matrix Mult - Takes two matrices and returns a new third matrix containing
    the dot product of the two matrices

"""

class Matrix:
    def __init__(self, col, row):
        self.con = []
        for r in range(row):
            entry = []
            for c in range(col):
                entry.append(0)
            self.con.append(entry)

    def mt_set (self, i, j, val):
        self.con[i][j] = val;
    
    def __str__(self):
        return self.con.__str__()

# Test
r = 10;
c = 10;
mat = Matrix (r,c);

for i in range (r):
    for j in range (c):
        mat.mt_set (i, j, i);

print (mat);

