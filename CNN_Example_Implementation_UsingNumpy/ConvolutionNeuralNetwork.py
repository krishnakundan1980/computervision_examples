# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:18:48 2018

@author: kkrishna
"""
import numpy as np
import pickle
import time
import scipy.misc as imgSaver

def SaveImage(frame, name, operation, index):
    fileName = ("%s_%s_%d.jpg" % (name, operation, index))
    imgSaver.imsave(fileName, frame)
    
def LogTimeStamp(remtime):
        hrs = int(remtime)/3600
        mins = int((remtime/60-hrs*60))
        secs = int(remtime-mins*60-hrs*3600)
        print("########  "+str(hrs)+"Hrs "+str(mins)+"Mins "+str(secs)+"Secs remaining  ########")

def ConvolveOperation(image, filt, bias):
    (l, w, w) = image.shape
    l1 = len(filt)
    ( _, f, f) = filt[0].shape
    w1 = w-f+1
    conv = np.zeros((l1,w1,w1))
    for jj in range(0,l1):
        for x in range(0,w1):
            for y in range(0,w1):
                conv[jj,x,y] = np.sum( image[: , x:x+f, y:y+f] * filt[jj] ) + bias[jj]
    conv[conv<=0] = 0
    return conv
    
def MaxPoolOperation(image, s1,s2):
    	(l, w, w) = image.shape
    	pooled_layer = np.zeros((l, int((w-s1)/s2+1),int((w-s1)/s2+1)))
    	for jj in range(0,l):
    		i=0
    		while(i<w):
    			j=0
    			while(j<w):
    				pooled_layer[jj,int(i/2),int(j/2)] = np.max(image[jj,i:i+s1,j:j+s1])
    				j+=s2
    			i+=s2
    	return pooled_layer
    
def GetFullyConnectedLayer(previous_layer_img, pooled_layer):
    (l, w, w) = previous_layer_img.shape
    return pooled_layer.reshape(int((w/2)*(w/2)*l),1)

 
class CNNModel(object):

    def __init__(self, img_width =28, img_height =28, img_depth =1, num_output = 10, filter_size = 5, 
                 lr = 0.01, conv1_num_filters = 8, conv2_num_filters = 8, conv3_num_filters = 8, mu = 0.95):
        self.IMG_WIDTH = img_width
        self.IMG_HEIGHT = img_height
        self.IMG_CHANNEL = img_depth  
        self.NUM_OUTPUT = num_output
        self.FILTER_SIZE = filter_size
        self.LEARNING_RATE = lr
        self.CONV1_NUM_FILT = conv1_num_filters
        self.CONV2_NUM_FILT = conv2_num_filters
        self.CONV3_NUM_FILT = conv3_num_filters
        self.MU = mu
        self.COST = []
        self.ACCURACY = []
        self.MODEL_NAME= 'trained_digit_detection_model.pickle'
        self.filt1 = {}
        self.filt2 = {}
        self.filt3 = {}
        self.bias1 = {}
        self.bias2 = {}
        self.bias3 = {}
        self.w1 = self.IMG_WIDTH - self.FILTER_SIZE + 1
        self.w2 = self.w1 - self.FILTER_SIZE + 1
        self.w3 = self.w2 - self.FILTER_SIZE + 1
        self.theta4 = {}
        self.bias4 = {}
        self.MODEL = [self.filt1, self.filt2, self.filt3, self.bias1, self.bias2, self.bias3, self.theta4, self.bias4, self.COST, self.ACCURACY]
        
    def Init_Filter_Using_Normalization(self, scale=1.0, distribution='normal'):
        if scale <= 0. or distribution.lower() not in {'normal'}:
            raise ValueError('`scale` or distribution argument {"normal" , "uniform"} invalid')
        
        seed_in = self.FILTER_SIZE*self.FILTER_SIZE*self.IMG_CHANNEL
        stddev = scale * np.sqrt(1./seed_in)
        shape = (self.IMG_CHANNEL,self.FILTER_SIZE,self.FILTER_SIZE)

        for i in range(0,self.CONV1_NUM_FILT):
            self.filt1[i] = np.random.normal(loc = 0,scale = stddev,size = shape)
            self.bias1[i] = 0
        
        seed_in = self.FILTER_SIZE*self.FILTER_SIZE*self.CONV1_NUM_FILT
        stddev = scale * np.sqrt(1./seed_in)
        shape = (self.CONV1_NUM_FILT,self.FILTER_SIZE,self.FILTER_SIZE)
        for i in range(0,self.CONV2_NUM_FILT):
            self.filt2[i] = np.random.normal(loc = 0,scale = stddev,size = shape)
            self.bias2[i] = 0
        
        seed_in = self.FILTER_SIZE*self.FILTER_SIZE*self.CONV2_NUM_FILT
        stddev = scale * np.sqrt(1./seed_in)
        shape = (self.CONV2_NUM_FILT,self.FILTER_SIZE,self.FILTER_SIZE)
        for i in range(0,self.CONV3_NUM_FILT):
            self.filt3[i] = np.random.normal(loc = 0,scale = stddev,size = shape)
            self.bias3[i] = 0
            
        self.theta4 = 0.01*np.random.rand(self.NUM_OUTPUT, int((self.w3/2)*(self.w3/2)*self.CONV3_NUM_FILT))
        self.bias4 = np.zeros((self.NUM_OUTPUT,1))

    def LoadTrainnedModel(self, model_name):
        self.MODEL_NAME = model_name
        pickle_in = open(model_name, 'rb')
        out = pickle.load(pickle_in)
        [self.filt1, self.filt2, self.filt3, self.bias1, self.bias2, self.bias3, self.theta4, self.bias4, self.COST, self.ACCURACY] = out
        self.MODEL = out
    
    def SaveTrainnedModel(self):
        with open(self.MODEL_NAME, 'wb') as file:
            pickle.dump([self.filt1, self.filt2, self.filt3, self.bias1, self.bias2, self.bias3, self.theta4, self.bias4, self.COST, self.ACCURACY], file)
        
    def Train(self, train_data, batch_size = 20, num_epochs = 2, learning_rate_mode = 'variation'):
        total_trainning_imgs = train_data.shape[0]
        learning_rate = self.LEARNING_RATE
        learning_constant_rate = True
        if learning_rate_mode.lower() in {'variation'}:
            learning_constant_rate = False
            
        #Start Training process
        for epoch in range(0, num_epochs):
            np.random.shuffle(train_data)
            x=0
            #Divide training data into batches
            batches = [train_data[k:k + batch_size] for k in range(0, total_trainning_imgs, batch_size)]
            for batch in batches:
                stime = time.time()
                if learning_constant_rate == False:
                    learning_rate =  learning_rate/(1+epoch/10.0)
                
                out = self.ApplyGradDescent(batch, learning_rate, self.IMG_WIDTH, self.IMG_CHANNEL, self.MU, self.filt1, self.filt2, self.filt3, self.bias1, self.bias2, self.bias3, self.theta4, self.bias4, self.COST, self.ACCURACY)
                [self.filt1, self.filt2, self.filt3, self.bias1, self.bias2, self.bias3, self.theta4, self.bias4, self.COST, self.ACCURACY] = out
                
                epoch_acc = round(np.sum(self.ACCURACY[int(epoch*total_trainning_imgs/batch_size):])/(x+1),2)
                
                #self.SaveTrainnedModel()
                
                per = float(x+1)/len(batches)*100
                print("Epoch:"+str(round(per,2))+"% Of "+str(epoch+1)+"/"+str(num_epochs)+", Cost:"+str(self.COST[-1])+", Batch_Acc:"+str(self.ACCURACY[-1]*100)+", Epoch_Acc:"+str(epoch_acc))
                ftime = time.time()
                batch_process_time = ftime-stime
                remaining_time = (len(batches)-x-1) * batch_process_time + batch_process_time*len(batches)*(num_epochs-epoch-1)
                LogTimeStamp(remaining_time)
                x+=1


    def ApplyGradDescent(self, batch, LEARNING_RATE, IMG_WIDTH, IMG_DEPTH, MU, filt1, filt2, filt3, bias1, bias2, bias3, theta4, bias4, cost, acc):
        X = batch[:,0:-1]
        X = X.reshape(len(batch),IMG_DEPTH,IMG_WIDTH,IMG_WIDTH)    
    
        y = batch[:,-1]
         
        #Init derivative variables
        n_correct=0
        cost_ = 0
        batch_size = len(batch)
        dfilt3 = {}
        dfilt2 = {}
        dfilt1 = {}
        dbias3 = {}
        dbias2 = {}
        dbias1 = {}
        v1 = {}
        v2 = {}
        v3 = {}
        bv1 = {}
        bv2 = {}
        bv3 = {}
        
        for k in range(0,len(filt1)):
            dfilt1[k] = np.zeros(filt1[0].shape)
            dbias1[k] = 0
            v1[k] = np.zeros(filt1[0].shape)
            bv1[k] = 0
    
        for k in range(0,len(filt2)):
            dfilt2[k] = np.zeros(filt2[0].shape)
            dbias2[k] = 0
            v2[k] = np.zeros(filt2[0].shape)
            bv2[k] = 0
        for k in range(0,len(filt3)):
            dfilt3[k] = np.zeros(filt3[0].shape)
            dbias3[k] = 0
            v3[k] = np.zeros(filt3[0].shape)
            bv3[k] = 0
        
        dtheta4 = np.zeros(theta4.shape)
        dbias4 = np.zeros(bias4.shape)
        v4 = np.zeros(theta4.shape)
        bv4 = np.zeros(bias4.shape)
            
        
        for i in range(0,batch_size):
            image = X[i]
            label = np.zeros((theta4.shape[0],1))
            label[int(y[i]),0] = 1
            ## Fetching gradient for the current parameters
            [dfilt1_, dfilt2_, dfilt3_, dbias1_, dbias2_, dbias3_, dtheta4_, dbias4_, curr_cost, acc_] = self.ForwardPropogation(image, label, filt1, filt2, filt3, bias1, bias2, bias3, theta4, bias4)
            for j in range(0,len(filt3)):
                dfilt3[j]+=dfilt3_[j]
                dbias3[j]+=dbias3_[j]
            for j in range(0,len(filt2)):
                dfilt2[j]+=dfilt2_[j]
                dbias2[j]+=dbias2_[j]
            for j in range(0,len(filt1)):
                dfilt1[j]+=dfilt1_[j]
                dbias1[j]+=dbias1_[j]
            dtheta4+=dtheta4_
            dbias4+=dbias4_
            cost_+=curr_cost
            n_correct+=acc_
            
        for j in range(0,len(filt1)):
            v1[j] = MU*v1[j] -LEARNING_RATE*dfilt1[j]/batch_size
            filt1[j] += v1[j]
            # filt1[j] -= LEARNING_RATE*dfilt1[j]/batch_size
            bv1[j] = MU*bv1[j] -LEARNING_RATE*dbias1[j]/batch_size
            bias1[j] += bv1[j]
        for j in range(0,len(filt2)):
            v2[j] = MU*v2[j] -LEARNING_RATE*dfilt2[j]/batch_size
            filt2[j] += v2[j]
            # filt2[j] += -LEARNING_RATE*dfilt2[j]/batch_size
            bv2[j] = MU*bv2[j] -LEARNING_RATE*dbias2[j]/batch_size
            bias2[j] += bv2[j]
        for j in range(0,len(filt3)):
            v3[j] = MU*v3[j] -LEARNING_RATE*dfilt3[j]/batch_size
            filt3[j] += v3[j]
            # filt2[j] += -LEARNING_RATE*dfilt2[j]/batch_size
            bv3[j] = MU*bv3[j] -LEARNING_RATE*dbias3[j]/batch_size
            bias3[j] += bv3[j]
            
        v4 = MU*v4 - LEARNING_RATE*dtheta4/batch_size
        theta4 += v4
        # theta3 += -LEARNING_RATE*dtheta3/batch_size
        bv4 = MU*bv4 -LEARNING_RATE*dbias4/batch_size
        bias4 += bv4
        
        cost_ = cost_/batch_size
        cost.append(cost_)
        accuracy = float(n_correct)/batch_size
        acc.append(accuracy)
        
        return [filt1, filt2, filt3, bias1, bias2, bias3, theta4, bias4, cost, acc]  
    
    ## Returns gradient for all the paramaters in each iteration
    def ForwardPropogation(self, image, label, filt1, filt2, filt3, bias1, bias2, bias3, theta4, bias4):
        conv1 = ConvolveOperation(image, filt1, bias1)
        conv2 = ConvolveOperation(conv1, filt2, bias2)
        conv3 = ConvolveOperation(conv2, filt3, bias3)
                
        ## Pooled layer with 2*2 size and stride 2,2
        pooled_layer = MaxPoolOperation(conv3, 2, 2)
        fc1 = GetFullyConnectedLayer(conv3, pooled_layer)
        ##Final Layer cost computation, WE can add more hidden layers here
        out = theta4.dot(fc1) + bias4	#10*1
        cost, probs = self.softmax_cost(out, label)
        
        if np.argmax(out)==np.argmax(label):
            acc=1
        else:
            acc=0
        
        return self.BackwardPropogation(image, conv1,filt1, conv2,filt2, conv3,filt3, fc1,theta4,label,probs,cost,acc)
      
    def BackwardPropogation(self, image, conv1, filt1, conv2, filt2, conv3, filt3, fc1, theta4, label, probs,cost,acc):
        (l, w, w) = image.shape	
        l1 = len(filt1)
        l2 = len(filt2)
        l3 = len(filt3)
        ( _, f, f) = filt1[0].shape
        w1 = w-f+1
        w2 = w1-f+1
        w3 = w2-f+1

        dout = probs - label
        dtheta4 = dout.dot(fc1.T)
        dbias4 = sum(dout.T).T.reshape((10,1))
        dfc1 = theta4.T.dot(dout)
        dpool = dfc1.T.reshape((l3, int(w3/2), int(w3/2)))
        dconv3 = np.zeros((l3, w3, w3))
        
        for jj in range(0,l3):
            i=0
            while(i<w3):
                j=0
                while(j<w3):
                    (a,b) = self.nanargmax(conv3[jj,i:i+2,j:j+2])
                    dconv3[jj,i+a,j+b] = dpool[jj,int(i/2),int(j/2)]
                    j+=2
                i+=2
        dconv3[conv3<=0]=0
        
        dconv2 = np.zeros((l2, w2, w2))
        dfilt3 = {}
        dbias3 = {}
        for xx in range(0,l3):
            dfilt3[xx] = np.zeros((l2,f,f))
            dbias3[xx] = 0
            
        dconv1 = np.zeros((l1, w1, w1))
        dfilt2 = {}
        dbias2 = {}
        for xx in range(0,l2):
            dfilt2[xx] = np.zeros((l1,f,f))
            dbias2[xx] = 0
        
        dfilt1 = {}
        dbias1 = {}
        for xx in range(0,l1):
            dfilt1[xx] = np.zeros((l,f,f))
            dbias1[xx] = 0
            
        for jj in range(0,l3):
            for x in range(0,w3):
                for y in range(0,w3):
                    dfilt3[jj]+=dconv3[jj,x,y]*conv2[:,x:x+f,y:y+f]
                    dconv2[:,x:x+f,y:y+f]+=dconv3[jj,x,y]*filt3[jj]
            dbias3[jj] = np.sum(dconv3[jj])
        
        dconv2[conv2<=0]=0
        
        for jj in range(0,l2):
            for x in range(0,w2):
                for y in range(0,w2):
                    dfilt2[jj]+=dconv2[jj,x,y]*dconv1[:,x:x+f,y:y+f]
                    dconv1[:,x:x+f,y:y+f]+=dconv2[jj,x,y]*filt2[jj]
            dbias2[jj] = np.sum(dconv2[jj])
        
        dconv1[conv1<=0]=0
        
        for jj in range(0,l1):
            for x in range(0,w1):
                for y in range(0,w1):
                    dfilt1[jj]+=dconv1[jj,x,y]*image[:,x:x+f,y:y+f]
            dbias1[jj] = np.sum(dconv1[jj])
    
        return [dfilt1, dfilt2, dfilt3, dbias1, dbias2, dbias3, dtheta4, dbias4, cost, acc]
    
    ## Predict class of each row of matrix X
    def Predict(self, image, label):
        filt1 = self.filt1
        filt2 = self.filt2
        filt3 = self.filt3
        bias1 = self.bias1
        bias2 = self.bias2
        bias3 = self.bias3
        theta4 = self.theta4
        bias4 = self.bias4
        
        (l,w,w)=image.shape
        (l1,f,f) = filt2[0].shape
        (l2,f,f) = filt3[0].shape
        l3 = len(filt3)
        w1 = w-f+1
        w2 = w1-f+1
        w3 = w2-f+1
        conv1 = np.zeros((l1,w1,w1))
        conv2 = np.zeros((l2,w2,w2))
        conv3 = np.zeros((l3,w3,w3))
        for jj in range(0,l1):
            for x in range(0,w1):
                for y in range(0,w1):
                    conv1[jj,x,y] = np.sum(image[:,x:x+f,y:y+f]*filt1[jj])+bias1[jj]
                    #SaveImage(conv1[jj], label, 'Conv1', jj)
        
        conv1[conv1<=0] = 0 
        #for jj in range(0,l1):
           # SaveImage(conv1[jj], label, 'ReLu', jj)
        
        for jj in range(0,l2):
            for x in range(0,w2):
                for y in range(0,w2):
                    conv2[jj,x,y] = np.sum(conv1[:,x:x+f,y:y+f]*filt2[jj])+bias2[jj]
                    #SaveImage(conv2[jj], label, 'Conv2', jj)
        
        conv2[conv2<=0] = 0
        #for jj in range(0,l2):
        #    SaveImage(conv2[jj], label, 'ReLu', jj)
        
        for jj in range(0,l3):
            for x in range(0,w3):
                for y in range(0,w3):
                    conv3[jj,x,y] = np.sum(conv2[:,x:x+f,y:y+f]*filt3[jj])+bias3[jj]
                    #SaveImage(conv3[jj], label, 'Conv3', jj)
        
        conv3[conv3<=0] = 0
        #for jj in range(0,l3):
            #SaveImage(conv3[jj], label, 'ReLu', jj)
        
        
        ## Pooled layer with 2*2 size and stride 2,2
        pooled_layer = MaxPoolOperation(conv3, 2, 2)
        #for jj in range(0,l3):
            #SaveImage(pooled_layer[jj], label, 'MaxPool', jj)
        
        fc1 = pooled_layer.reshape((int((w3/2)*(w3/2)*l3),1))
        out = theta4.dot(fc1) + bias4	#10*1
        eout = np.exp(out, dtype=np.float)
        probs = eout/sum(eout)
        return np.argmax(probs), np.max(probs)
    
    ## Returns idexes of maximum value of the array
    def nanargmax(self, a):
    	idx = np.argmax(a, axis=None)
    	multi_idx = np.unravel_index(idx, a.shape)
    	if np.isnan(a[multi_idx]):
    		nan_count = np.sum(np.isnan(a))
    		# In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
    		idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
    		multi_idx = np.unravel_index(idx, a.shape)
    	return multi_idx
    
    ## Returns cost and probablity for the given output w.r.t. actual known truth values
    def softmax_cost(self, out,y):
    	eout = np.exp(out, dtype=np.float)
    	probs = eout/sum(eout)
    	p = sum(y*probs)
    	cost = -np.log(p)
    	return cost,probs
    
    def PrintLearnedFilters(self):	
        for jj in range(0, len(self.filt1)):
            SaveImage(self.filt1[jj][0], '', 'Filter1', jj)
        for jj in range(0, len(self.filt2)):
            SaveImage(self.filt2[jj][0], '', 'Filte2', jj)
        for jj in range(0, len(self.filt3)):
            SaveImage(self.filt3[jj][0], '', 'Filter3', jj)
        

      
      
