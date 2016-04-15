# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from os import listdir
from os.path import isfile, join
import cv2
# display plots in this notebook



# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap



import sys
caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

import os

'''
Take a vector as an input and image as key and stores in a CSV file.
@param key:		store product key to which the vector belongs.
@param vector: 	To be stored in csv file
@param flag:	a - to append to file
				w - to overwrite
@param csvFile: file in which the vector needs to be added.
'''
def vec2csv(key, vector, flag, csvFile):
	df = pd.DataFrame(data=[[key, vector]], columns=["key","vector"])
		
	if flag =="a":
		dfile = pd.read_csv(csvFile, index_col=0)
		df = dfile.append(df)
		
	df.to_csv(csvFile)


#print sys.argv[1]

#caffe.set_mode_cpu()
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()



model_def = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST) 
                
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR



# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
                          
                          
lists=[]         
index=[]                 

myBikePath= caffe_root + 'examples/images/Myntra/Tshirts/Train'
onlyfiles = [ f for f in listdir(myBikePath) if isfile(join(myBikePath,f)) ]

#Getting Descriptors 
print "Training the KNN"
dict = {}
count=0
image_classes=[]
im_feat=np.zeros((len(onlyfiles),4096),dtype=np.float32)
for n in range(0, len(onlyfiles)):

    image = caffe.io.load_image(join(myBikePath,onlyfiles[n]))
    transformed_image = transformer.preprocess('data', image)
   # plt.imshow(image)                                          


# copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

### perform classification
    output = net.forward()

    #output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

   # print 'predicted class is:', output_prob.argmax()
    feat = net.blobs['fc7'].data[0]
    
    print count,"-->",onlyfiles[n]
       
    feat=feat.reshape(1,4096)
    k=np.array(feat,dtype=np.float32)
    #dict[onlyfiles[n]] = feat
    im_feat[n]=k
    image_classes+=[count]
    count=count+1
    #if n==0:
     #  vec2csv(join(myBikePath,onlyfiles[n]),feat,"a","data.csv")
    #else:
     #  vec2csv(join(myBikePath,onlyfiles[n]),feat,"a","data.csv")          
feat
print "Training K Nearest"

#Initailzing kNN
knn = cv2.KNearest()

#Training KNN
knn.train(im_feat,np.array(image_classes))


myTestPath= caffe_root + 'examples/images/Myntra/Tshirts/Test'
onlyfiles = [ f for f in listdir(myTestPath) if isfile(join(myTestPath,f)) ]

#Getting Descriptors 
print "----------------Results-----------"

im_feat=np.zeros((len(onlyfiles),4096),dtype=np.float32)
for n in range(0, len(onlyfiles)):

    image = caffe.io.load_image(join(myTestPath,onlyfiles[n]))
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    feat = net.blobs['fc7'].data[0]
    #print onlyfiles[n]
    #print count
    feat=feat.reshape(1,4096)
    k=np.array(feat,dtype=np.float32)
    ret, results, neighbours ,dist = knn.find_nearest(k, 1)  
    print ret," ",onlyfiles[n]     



'''
     
image = caffe.io.load_image(caffe_root + 'examples/images/test2.jpg')
transformed_image = transformer.preprocess('data', image)
   # plt.imshow(image)                                          


# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

    #output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

   # print 'predicted class is:', output_prob.argmax()
feat = net.blobs['fc7'].data[0]  
feat=feat.reshape(1,4096) 
k=np.array(feat,dtype=np.float32)
ret, results, neighbours ,dist = knn.find_nearest(k, 1)  

print ret," --test2"


     
image = caffe.io.load_image(caffe_root + 'examples/images/test3.jpg')
transformed_image = transformer.preprocess('data', image)
   # plt.imshow(image)                                          


# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

    #output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

   # print 'predicted class is:', output_prob.argmax()
feat = net.blobs['fc7'].data[0]  
feat=feat.reshape(1,4096) 
k=np.array(feat,dtype=np.float32)
ret, results, neighbours ,dist = knn.find_nearest(k, 1)  

print ret," --test3"


     
image = caffe.io.load_image(caffe_root + 'examples/images/test4.jpg')
transformed_image = transformer.preprocess('data', image)
   # plt.imshow(image)                                          


# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

    #output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

   # print 'predicted class is:', output_prob.argmax()
feat = net.blobs['fc7'].data[0]  
feat=feat.reshape(1,4096) 
k=np.array(feat,dtype=np.float32)
ret, results, neighbours ,dist = knn.find_nearest(k, 1)  

print ret," --test4"


     
image = caffe.io.load_image(caffe_root + 'examples/images/test7.jpg')
transformed_image = transformer.preprocess('data', image)
   # plt.imshow(image)                                          


# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

    #output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

   # print 'predicted class is:', output_prob.argmax()
feat = net.blobs['fc7'].data[0]  
feat=feat.reshape(1,4096) 
k=np.array(feat,dtype=np.float32)
ret, results, neighbours ,dist = knn.find_nearest(k, 1)  

print ret," --test7"


#df = pd.DataFrame(dict,index=0) 
#print df
    
  #  print type(feat)
   # lists.append(feat)
    #index.append(sys.argv[i])
#my_df = pd.DataFrame({"DATA":lists})

#my_df.to_csv('my_csv.csv', index=False, header=False)
#Z.tofile('foo.csv',sep=',',format='%10.5f')
np.savetxt("foo1.csv", im_feat, delimiter=",")


'''






