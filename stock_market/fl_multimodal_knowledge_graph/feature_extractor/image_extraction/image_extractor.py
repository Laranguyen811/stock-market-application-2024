from PIL import Image

import numpy as np
import PIL as image
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath,to_2tuple, trunc_normal_
from timm.models.registry import register_model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
import numpy as np
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools import download_images

TF_ENABLE_ONEDNN_OPTS=0
def load(path: str) -> np.array:
    ''' Takes an image from the given path, resizes to 224x224 and returns a numpy array of float32 of the image.
    Inputs:
        path(string): A string of the path of the image to load.
    Returns:
        np.array: A numpy array of the resized image with dtype float32.
    Raises:
        Exception: Throwing an exception if the error loading image
    '''
    try: # Trying a block of code before throwing an exception
        img = Image.open(path).resize(224,224),Image.BICUBIC()  # Resizing the image using bicubic interpolation (method providing smoother and higher quality resized images)
        img = np.asarray(img).astype(np.float32)  # Converting the image into a 2D array of float32
        img /= 255.0  # Normalising pixel values to [0,1]
        return img
    except Exception as e:  #Throwing an Exception as variable e
        print("Error loading image: {e}")
        return None


# Initializing TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')  # Creating a TPUClusterResolver object which identifies and connects to TPU cluster, with default tpu address.
tf.config.experimental_connect_to_cluster(resolver)  # Connecting TPU runtime to cluster with specified resolver, setting up communication between our code and the TPU
tf.tpu.experimental.initialize_tpu_system(resolver)  # Initializing the TPU system, performing any needed configuration and setup for it to be ready for use

# Creating a TPU strategy
strategy = tf.distribute.experimental.TPUStrategy(resolver)  # Creating a TPU strategy object for distributing the training across TPU devices, handling any distribution of data and model across the TPU cores.

# Splitting the data into the training set, the validation set and the test set
X_train, y_train = train_test_split(download_images.download_images(text='',save_path=''), test_size=0.2, random_state=42)

# Creating a StratifiedGroupKfold object
sgkf = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)

class SparseAutoencoder:
    def __init__(self,input_dim, encoding_dim):
        super(SparseAutoencoder,self).__init__()
        # Initializing TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu='')  # Creating a TPUClusterResolver object which identifies and connects to TPU cluster, with default tpu address.
        tf.config.experimental_connect_to_cluster(
            resolver)  # Connecting TPU runtime to cluster with specified resolver, setting up communication between our code and the TPU
        tf.tpu.experimental.initialize_tpu_system(
            resolver)  # Initializing the TPU system, performing any needed configuration and setup for it to be ready for use

        # Creating a TPU strategy
        strategy = tf.distribute.experimental.TPUStrategy(resolver)  # Creating a TPU strategy object for distributing the training across TPU devices, handling any distribution of data and model across the TPU cores.
        with strategy.scope():
            self.encoder = Sequential([  # Creating a sequential model
                Dense(128, activation='relu', input_shape=(input_dim,)), # Adding a dense (fully connected) layer as an encoding layer with 128 units and ReLU activation. The input shape is a vector length of input dimensions
                Dense(input_dim,activation='sigmoid')  # Adding a dense layer as a bottleneck layer with input dimensions as sigmoid as an activation function
            ])
            self.decoder = Sequential([
                Dense(128, activation='relu',input_shape=(encoding_dim,)),
                Dense(input_dim,activation='sigmoid')# Adding a dense layer as a bottleneck with input-dimension units and a sigmoid activation, used for reconstruction
            ])
            self.decoder = Sequential([  # Creating a sequential model
                Dense(128, activation='relu',input_shape=(encoding_dim,)),  # Adding a dense (fully connected) layer as an decoding layer with 128 units and ReLU activation. The input shape is a vector length of input dimensions
                Dense(input_dim,activation='sigmoid')
            ])
            self.autoencoder = Sequential([self.encoder,self.decoder])
            self.autoencoder.compile(optimizer='adam',loss='cross-entropy',metrics='accuracy') # Specifying optimizer as Adam optimizer, the loss function as cross-entropy (used for multi-class classification), the metric that we are tracking is accuracy

def kl_divergence(p,p_hat):
    ''' Takes p and p_hat and returns KL divergence.
    Inputs:
        p (int): An integer of p value
        p_hat (int): An integer of p_hat value
    Returns:
    float: a float value of KL divergence
    '''
    epsilon = 1e-10  # Adding a small value to avoid division by zero
    return p * tf.math.log(p/p_hat) + (1 - p) * tf.math.log((1-p)/(1-p_hat))

#def process_model(use_data_parallel: bool = False, device_ids: list = None) -> torch.nn.Models:
