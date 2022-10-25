import cv2 
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


def lagrange(x ,x_ar ,y_ar,length ):
    b = np.multiply(np.array(y_ar, dtype=np.float64),np.full((1,length),-1,np.float64))#np.add(np.multiply(np.array(y_ar),np.full((1,length),-1)),np.full((1,length),y[length])) # b = [y]
    Li = []
    for o,oo in enumerate(x) :
        sum = 0 
        for i in range(length):
            a = np.array(x_ar, dtype=np.float64) #  a = [x]
            temp1 = np.subtract(np.full((1,length),a[i],np.float64),a) #temp = [x_0] - [x]
            temp1 = reduce(lambda a,b : a*b, temp1[temp1!=0])
            temp2 = np.subtract(np.full((1,length),oo),a) #temp = x - [x]
            temp2 = np.append(temp2[:,0:i],temp2[:,i+1:length])
            temp2 = reduce(lambda a,b: a*b ,temp2)
            L = temp2/temp1
            L *= b[:,i] ; sum += L
        Li += [sum]
        #print(Li)
    return Li

def recur(table ,x_arr,x,y):
    if x==y :
        if table[x][x] == 999999999 :
            recur(table , x_arr , x,y+1)
        else :
            recur(table,x_arr,x+1,y+1)
    else :
        


def newton(x ,x_ar ,y_ar,length ,table):



    return 

def makeMat(n,x):
    b = np.array(np.full((1,n),-1))
    for i in range(n-2):
        b = np.vstack((np.full((1,n),999999999),b))
    b = np.vstack((x,b))
    print(b)
    return b

up_path = "C:\\Users\\User\\Documents\\L.S\\numerical analysis\\airplane_3\\up_binary.png"
img = cv2.imread(up_path,cv2.IMREAD_GRAYSCALE )
down__path = "C:\\Users\\User\\Documents\\L.S\\numerical analysis\\airplane_3\\up_binary.png"
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
    if i % 10 == 0 :
        x += [i/10]
        y += [j/10]

x = list(map(lambda i : i - x[0] , x)) #左移
x_res = np.arange(x[0],x[len(x)-1]+1)
y_res = lagrange(x_res, x,y,len(x))
#plt.plot(x, y)
plt.plot(x_res, y_res)


##########down##########
x = [] ; y = []
for i,j  in (down_plane) : # mapping x -> y 
    if i % 10 == 0 :
        x += [i/10]
        y += [j/10+20]
x_res = np.arange(x[0],x[len(x)-1]+1)
y_res = lagrange(x_res, x,y,len(x))
#plt.plot(x, y)
plt.plot(x_res, y_res)



makeMat(5,[1,2,3,4,5])
plt.show()
print("HELLO")
