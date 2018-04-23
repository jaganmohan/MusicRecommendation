import numpy as np
#from numba import vectorize
import os

def createdata():
    path = 'matrix.csv'
    if (os.path.exists(path)):
        matrix = np.genfromtxt(path, delimiter=',')
        # mask = np.genfromtxt('mask.csv',delimiter=',')
        # mask = mask.astype(int)
        # matrix = matrix.T
        # matrix = matrix[mask]
        # matrix = matrix.T
    
    #matrix = scipy.sparse.rand(N, N, 1, format='csr')
    else:
        def randintgen():
            return np.random.randint(50,100,size=(100,100))
        #matrix = np.zeros((1000,1000))
        matrix = np.random.choice(3, (1000,1000),p=[0.7, 0.2, 0.1])
        for i in range(10):
            matrix[i*100:(i+1)*100,i*100:(i+1)*100]=randintgen()
        np.savetxt("matrix.csv", matrix, delimiter=",",fmt='%2.0f')
    print (matrix.shape)
    return matrix



def createconfidence(Matrix):
    alpha = 2.0
    epsilon = 1e-6
    Matrix = alpha* np.log(1+Matrix/epsilon)
    return Matrix
        
def factorize(Matrix):
    """Matriz: sparse matrix to be factorized using WMF
    """
    no_users, no_items = Matrix.shape
    MT = Matrix.T
    no_factors = 50
    U = None
    V = np.random.randn(no_items, no_factors).astype(np.float32)*0.01
    
    for i in range(30):
        #print (i)
        U=computefactors(V,Matrix,no_users, no_items,no_factors)
        
        V=computefactors(U,MT, no_items,no_users,no_factors)
        
    return U,V    
        
#@vectorize(["float32(float32,float32,int32,int32,int32)"],target='cuda')       
def computefactors(Y, Matrix,m,n,f):
    """
    compute the matrix Y 
    Y - can be user or item matrix
    """
 #f is no of factors remain same for both user and item input
    YTY = np.dot(Y.T,Y)
    lambda1 = 1e-8
    #lambda1=0
    YTY += lambda1*np.eye(f)
    
    X_new = np.zeros((m,f),dtype = np.float32)
   
    for i in range(m):
        C= Matrix[i]
        #Cu = np.diag(C)
        #S = Cu-np.eye(n)
        #YTSY = np.dot(Y.T,np.dot(S,Y))
        A= YTY#+YTSY
        #P = np.zeros_like(C)
        #P[C>0]=1
        #B = np.dot(np.dot(Y.T,Cu),P)
        B= np.dot(C,Y)
        x = np.linalg.solve(A,B)
        X_new[i] = x
        
        
    return X_new
    

if __name__ =='__main__':
    matrix = createdata()
    #matrix = createconfidence(matrix)
    np.savetxt('cmatrix.csv',matrix,delimiter = ',')
    U,V = factorize(matrix)
    np.savetxt("U.csv", U, delimiter=",")#,fmt='%2.0d')
    np.savetxt("V.csv", V, delimiter=",")#,fmt='%2.0d')

