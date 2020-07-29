import os 
from PIL import Image
import numpy as np
from array import *

with open('./sample_train.txt') as f:
    lines = f.read().splitlines()

mlist = []

for i in lines:
    words=list(i.split())  # Extract one line
    mlist.append(words)   # Now all files name( with name) is contained in the mlist

mlist.sort()

mat  = np.matrix(mlist)
x 	 = mat[:,0:1]
x  	 = np.array(x).flatten()

mat  = np.matrix(mlist)
y 	 = mat[:,1:2]
y  	 = np.array(y).flatten() # Contain sample names

dic = {
}

j=1;
for i in y:
	dic[i] = j
	j+=1;

# for x in dic:
#   print(dic[x])

  #Alice : 3 ,  Bob : 6 , ABC : 9

im_matrix = np.empty((10,1024))

j=0
for i in x:	
	img = np.array(Image.open(i).convert('L')) # Converting all images into grayscale & images to array
	img_r = np.resize(img,(32,32)) # Resize image
	pix = img_r.flatten()
	im_matrix[j] = pix  # Now creating a matrix to store all flatten images
	j+=1


im_mean = im_matrix.mean(0)  # 0 refers to column wise mean
im_matrix = im_matrix - im_mean    # mean matrix
 
u,s,v = np.linalg.svd(im_matrix)

v = np.transpose(v); # Transpose
coeff = np.matmul(im_matrix,v)     #Coefficient Matrix

##
with open('./sample_test.txt') as f:
    lines = f.read()

mlist = []

for i in lines:
    words=list(i.strip())  # Extract one line
    mlist.append(words)   # Now all files name( with name) is contained in the mlist

mat  = np.matrix(mlist)
x 	 = mat[:,0:1]
x  	 = np.array(x).flatten()

im_matrix = np.empty((10,1024))

j=0
for i in x:	
	img = np.array(Image.open(i).convert('L')) # Converting all images into grayscale & images to array
	img_r = np.resize(img,(32,32)) # Resize image
	pix = img_r.flatten()
	im_matrix[j] = pix  # Now creating a matrix to store all flatten images
	j+=1

im_matrix = im_matrix - im_mean    # mean matrix
coeff = np.matmul(im_matrix,v)     #Coefficient Matrix
##


mea = {}
var = {}

j=1;
for i in y:
	dic[i] = j
	j+=1;


first = 0
mean={}
var={}
for i in sorted(dic, key=dic.get):
	last = dic[i]
	mean[i] = np.mean(coeff[first:last, :], axis=0)
	var[i] = np.var(coeff[first:last, :], axis=0)
	first = last

#x   from the Sample Image
mu = im_matrix  
sigma = coeff

e = 2.714
pi = 3.142

maxi = -1;
samples = len(mean)

for i in coeff:
	for j in  dic:
		m_dist_x = ((i - mean[j])**2/(2*var[j]**2))
		m_dist_x = (e**(-1)*m_dist_x)
		m_dist_x = 1/sqrt(2*pi*var[j]**2)
		p = np.prod(m_dist_x)
		if(maxi < p):
			maxi = p
			max_class = j

	print (max_class)

#	maxi2.append(maxi)	
