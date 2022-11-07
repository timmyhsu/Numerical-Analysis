import cv2 
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


def lagrange_poly(x_ar ,y_ar) :
    Li = np.poly1d([])
    for (i,j) in enumerate(x_ar) :
        #print(x_ar[0:i]+x_ar[i+1:])
        temp1 = np.poly1d(x_ar[0:i]+x_ar[i+1:],True) #(x-x_i)連乘
        temp2 = list(map(lambda a : j-a,x_ar[0:i]+x_ar[i+1:])) #Li分母，(x_i-x_j)先減
        temp2 = reduce(lambda a,b : a*b,temp2)#Li分母，連乘
        tempLi = temp1/temp2 
        tempLi*= y_ar[i]
        Li += tempLi
    return Li




up_path = "C:\\Users\\timmy\\Documents\\Coding_Plarground\\vscode\\Numerical-Analysis\\airplane_4\\binary.png"
img = cv2.imread(up_path,cv2.IMREAD_GRAYSCALE )
down__path = "C:\\Users\\timmy\\Documents\\Coding_Plarground\\vscode\\Numerical-Analysis\\airplane_4\\binary.png"
#cv2.imshow("windows",img)
img2 = cv2.imread(down__path,cv2.IMREAD_GRAYSCALE )
#cv2.waitKey(0)


up_plane=[]
down_plane=[]
#print(img.shape)

###########find the edge ###########
(height,width) = img.shape

for x in range(width):
    for y in range(height):
        if img[y][x] == 0 :
            up_plane += [[x,y]] # from y = 0 to height , find the upper edge 
            break
        
(height,width) = img2.shape
print(img2.shape)
for x in range(width):
    for y in range(height-1,0,-1):
        if img2[y][x] == 0 : 
            down_plane += [[x,y]] # from y = height to 0 , find the down edge 
            break

#########up###########
x = [] ; y = []
for i,j  in (up_plane) : # mapping x -> y 
    if i % 20 == 0 :
        x += [i/10]
        y += [(-j+250)]

x = list(map(lambda i : i - x[0] , x)) #左移

print("uppoints:",x,y,sep='\n')

print("up:",lagrange_poly(x,y))
x_res = np.arange(100) 
y_res = lagrange_poly(x,y)(x_res)

plt.plot(x_res,y_res)

##########down##########
x = [] ; y = []
for i,j  in (down_plane) : # mapping x -> y 
    if i % 20 == 0 :
        x += [i/10]
        y += [(-j+250)/10]
        
print("uppoints:",x,y,sep='\n')
print("down:",lagrange_poly(x,y))
#print(x,y)
x = list(map(lambda i : i - x[0] , x)) #左移
x_res = np.linspace(0,115,1)
y_res = lagrange_poly(x,y)(x_res)
plt.plot(x_res,y_res)





plt.show()
print("HELLO")
