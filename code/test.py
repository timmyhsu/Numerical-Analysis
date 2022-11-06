import numpy as np 
from functools import reduce

x_ar = [1.3,1.6,1.9]
y_ar = [3,4,5,6]
temp = np.poly1d(x_ar)
for (i,j) in enumerate(x_ar) :
    print(x_ar[0:i]+x_ar[i+1:])
    temp1 = np.poly1d(x_ar[0:i]+x_ar[i+1:],True) #Lagrange
    temp2 = list(map(lambda a : j-a,x_ar[0:i]+x_ar[i+1:])) 
    temp2 = reduce(lambda a,b : a*b,temp2)
    tempLi = temp1/temp2
    print(temp1,"\n",temp2,"\n",tempLi,"\n",tempLi.deriv())
#print(np.poly1d([1])-2*np.poly1d([1,-1]))
print(temp.deriv()(1))
