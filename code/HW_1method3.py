import cv2 
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


up_path = "C:\\Users\\timmy\\Documents\\Coding_Plarground\\vscode\\Numerical-Analysis\\airplane_4\\binary.png"
img = cv2.imread(up_path,cv2.IMREAD_GRAYSCALE )
down__path = "C:\\Users\\timmy\\Documents\\Coding_Plarground\\vscode\\Numerical-Analysis\\airplane_4\\binary.png"
#cv2.imshow("windows",img)
img2 = cv2.imread(down__path,cv2.IMREAD_GRAYSCALE )
#cv2.waitKey(0)

def func(x_ar,y_ar):
    res = []
    enumer = list(enumerate(x_ar))[1:]
    for (i,j) in enumer:
        if i % 10 == 0 :
            m = (y_ar[i]-y_ar[i-10])/(x_ar[i]-x_ar[i-10])
            plt.plot([x_ar[i],x_ar[i-10]],[y_ar[i],y_ar[i-10]])
            k = y_ar[i] - m*x_ar[i]
            res += [[m,k]]
    return res 
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
    if i % 1 == 0 :
        x += [i/10]
        y += [(-j+250)/10]

x = list(map(lambda i : i - x[0] , x)) #左移
print(func(x,y))
#plt.plot(x,y)

##########down##########
x = [] ; y = []
for i,j  in (down_plane) : # mapping x -> y 
    if i % 1 == 0 :
        x += [i/10]
        y += [(-j+250)/10]
x = list(map(lambda i : i - x[0] , x)) #左移
plt.plot([x[len(x)-10],x[len(x)-1]],[y[len(x)-10],y[len(x)-1]])
x = list(map(lambda i : i - x[0] , x)) #左移
print(func(x,y))





plt.show()
print("HELLO")
