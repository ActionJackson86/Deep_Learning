#!/usr/bin/env python
# coding: utf-8

# In[1]:


#State Farm Distracted Driver Detection
#Deep Learning - CSC570
#Jackson Roach

from keras.utils import to_categorical, multi_gpu_model
from keras.models import Model
from keras.layers import Dense, Input, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import pandas as pd
import sys
from PIL import Image
import os

sys.modules['Image'] = Image


# In[2]:


train_path = '/home/ubuntu/final/imgs/train/'
test_path = '/home/ubuntu/final/imgs/test/'   
driver_imgs_csv = pd.read_csv('/home/ubuntu/final/driver_imgs_list.csv')


# In[3]:


#load data
train_data_generation = ImageDataGenerator()
test_data_generation = ImageDataGenerator()

train_set = train_data_generation.flow_from_directory(train_path, target_size = (32, 32), color_mode = "rgb", batch_size = 4, class_mode = 'categorical', shuffle = True, seed = 42)

test_set = test_data_generation.flow_from_directory(test_path, 
                                                    target_size = (32, 32),
                                                    color_mode = "rgb",
                                                    batch_size = 4,
                                                    class_mode = 'categorical',
                                                    shuffle = False,
                                                    seed = 42)


# In[4]:


def build_network(num_gpu = 1, input_shape = None):
    inputs = Input(shape = input_shape, name = "input")
    
    #block 1
    conv1 = Conv2D(64, (3,3), activation = "relu", name = "conv_1")(inputs)
    batch1 = BatchNormalization(name = "batch_norm_1")(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2), name = "pool_1")(batch1)
    
    #block 2
    conv2 = Conv2D(64, (3,3), activation = "relu", name = "conv_2")(pool1)
    batch2 = BatchNormalization(name = "batch_norm_2")(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2), name = "pool_2")(batch2)
    
    #fully connected layers
    flatten = Flatten()(pool2)
    fc1 = Dense(512, activation = "relu", name = "fc1")(flatten)
    d1 = Dropout(rate = 0.2, name = "dropout1")(fc1)
    fc2 = Dense(256, activation = "relu", name = "fc2")(d1)
    d2 = Dropout(rate = 0.2, name = "dropout2")(fc2)
    
    #output
    output = Dense(10, activation = "softmax", name = "softmax")(d2)
    
    #compile
    model = Model(inputs = inputs, outputs = output)
    if num_gpu > 1:
        model = multi_gpu_model(model, num_gpu)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model


# In[6]:


def main():
    
    #build model
    model = build_network(num_gpu = 1, input_shape = (32, 32, 3))
    
    #fit model
    model.fit_generator(train_set,
                    steps_per_epoch = 22400/4,
                    epochs = 5,
                    validation_data = test_set,
                    validation_steps = 4000/4)
if __name__ == "__main__":
    main()


# In[ ]:




