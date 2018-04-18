import numpy as np
U = np.genfromtxt('U.csv', delimiter = ',')
V = np.genfromtxt('V.csv', delimiter = ',')

P = np.dot(U,V.T)
np.savetxt('P.csv',P , delimiter = ',', fmt = "%f")