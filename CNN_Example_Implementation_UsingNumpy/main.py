# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:39:10 2018

@author: kkrishna
"""

import numpy as np
from ConvolutionNeuralNetwork import *
import gzip

def ExtractData(filename, img_channel, img_width, img_height, num_images):
    print('Extracting file..', filename)
    with gzip.open(filename) as bytesstream:
        bytesstream.read(16)
        data_buff = bytesstream.read(img_channel*img_width*img_height*num_images)
        data = np.frombuffer(data_buff, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, img_channel*img_width*img_height)
        return data
    
def ExtractLabels(filename,img_channel,num_images):
    print('Extracting file..', filename)
    with gzip.open(filename) as filestream:
        filestream.read(8)
        data_buff = filestream.read(img_channel*num_images)
        data = np.frombuffer(data_buff, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images,img_channel)
        return data

def main():
    #Set CNN Hyper parameters, This network consist of following layers - Input, Conv1, Relu1, Conv2, Relu2, Pooling, FC1 and output
    IMG_WIDTH = 28
    IMG_HEIGHT = 28
    IMG_CHANNEL = 1    
    
#    print("Load MNIST zip data set from 'http://yann.lecun.com/exdb/mnist/' ....This include training and test datasets...")    
    X = ExtractData('train-images-idx3-ubyte.gz', IMG_CHANNEL, IMG_WIDTH, IMG_HEIGHT, 50000)
    y_dash = ExtractLabels('train-labels-idx1-ubyte.gz', IMG_CHANNEL, 50000)
#        
#    #Bring data in a normalized form for the training use purpose
    X = X - X.mean()
    X = X/X.std()
    training_data = np.hstack((X,y_dash))
    np.random.shuffle(training_data)    
    
    #Train the model after Init of Hyper parameters   
    net = CNNModel(filter_size = 5, conv1_num_filters = 12, conv2_num_filters = 12, conv3_num_filters = 12)
    net.Init_Filter_Using_Normalization()
    #net.LoadTrainnedModel('trained_digit_detection_model.pickle')
    net.Train(training_data, batch_size = 20, num_epochs = 2, learning_rate_mode='constant')
    net.SaveTrainnedModel()
    
    #print leanred filters
    net.PrintLearnedFilters()

if __name__ == '__main__':
    main()