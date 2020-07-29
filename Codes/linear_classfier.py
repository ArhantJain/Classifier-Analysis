import matplotlib.image as mpimg
import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dirListing = os.listdir('./dataset') # Access the directory
editFiles = []
for item in dirListing:
   if ".jpg" in item:
       editFiles.append('./dataset' +'/'+item) # Saving path of images in list

#print (editFiles)

im_matrix = np.empty((520,1024))
j=0
for i in editFiles:	
	img = np.array(Image.open(i).convert('L')) # Converting all images into grayscale & images to array
	img_r = np.resize(img,(32,32)) # Resize image
	pix = img_r.flatten()
	im_matrix[j] = pix  # Now creating a matrix to store all flatten images
	j+=1



num_data,dim = im_matrix.shape # Gives Rows and columns

im_mean = im_matrix.mean(0)
im_matrix = im_matrix - im_mean

u,s,v = np.linalg.svd(im_matrix) # S: Gives eigen value , V : Set of eigen vector

x = np.sum(s) # Sum of all eigen values
s = np.sort(s)

#print (x)
#print (s)
#print (v.shape)

su = 0
x_axis = []
y_axis = []

for i in range(0,520,1):
	su += s[i];
	per = su/x;  	   # Percentage of image re-constructed
	x_axis.append(i+1) # x-axis contains : No. of feature taken for the reconstruction of image
	y_axis.append(per)

v = np.transpose(v); # Transpose
coeff = np.matmul(im_matrix,v) #Coefficient Matrix , Reconstruction

a = coeff[:,0:1]
b = coeff[:,1:2]
c = coeff[:,2:3]
d = np.zeros(520)

plt.scatter(a,d)
plt.xlabel('PCA-1')
plt.title('1-Dimensinal Plot')

plt.figure()
plt.scatter(a,b)
plt.xlabel('PCA-2')
plt.ylabel('PCA-3')
plt.title('2-Dimensinal Plot')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(a, b, c, c='r', marker='o')

ax.set_xlabel('PCA-1')
ax.set_ylabel('PCA-2')
ax.set_zlabel('PCA-3')
plt.title('3-Dimensinal Plot')

plt.figure()
plt.plot(x_axis, y_axis)

plt.xlabel('No. of images taken')
plt.ylabel('Amount of image reconstructed') 
plt.title('Image Reconstruction Plot')
 
# function to show the plot

plt.show()


