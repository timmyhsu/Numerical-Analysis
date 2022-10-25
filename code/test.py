import numpy as np 
from functools import reduce

a = np.array([4,5,6])
b = np.array([1,2,3])
#b = reduce(lambda a , b : a*b ,b)
c = np.vstack((a,b))
print(c[1][1])