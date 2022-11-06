import cv2 
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

x  = [-8.2947705,-7.018728,4.33613599,-2.3654881,7.63990620]
y  = [-0.7033667,0.0977579,0.6878709,0.4335587,3.6101307]
y_ = [1.0907,0.3125,0.2491176,0.01639344,1.1594202]

def hermitpoly(x_ar , y_ar , y_):
    res = np.poly1d([0])
    for (i,j) in enumerate(x_ar) :
        #print(x_ar[0:i]+x_ar[i+1:])
        temp1 = np.poly1d(x_ar[0:i]+x_ar[i+1:],True) #Lagrange
        temp2 = list(map(lambda a : j-a,x_ar[0:i]+x_ar[i+1:])) 
        temp2 = reduce(lambda a,b : a*b,temp2)
        tempLi = temp1/temp2
        H = (np.poly1d([1])-2*np.poly1d([1,-j])*(tempLi.deriv()(j)))*(tempLi**2)
        H *= y_ar[i]
        H_ = np.poly1d([j],True)*(tempLi**2)*y_[i]
        res = H + H_
    print(res)
    return res

x_res = np.linspace(-10,10,1)
y_res = hermitpoly(x,y,y_)(x_res)
plt.plot(x_res,y_res)
plt.show()
print("HELLO")