import tensorflow
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
import keras_tuner as kt
from tensorflow.keras import layers
import keras
import time as t
from tensorflow.keras.preprocessing import image
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

img_arrays = image.img_to_array(download_images.download_images(text='',save_path=''))
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_img_arrays = img_arrays.map(lambda x,y: (normalization_layer(x),y))
# Splitting the data into the training set, the validation set and the test set
X_train, X_test, y_train,y_test = train_test_split(normalized_img_arrays, test_size=0.2, random_state=42)

# Creating a StratifiedGroupKfold object
sgkf = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)


def build_model(hp):
    model = keras.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))

    # Encoder
    model.add(layers.Dense(units=hp.Int('units', min_value=31, max_value=512,step=32), activation='relu',
                           kernel_initializer=hp.Choice('kernel_initializer', ['glorot_uniform', 'he_uniform'])  # Glorot Uniform Initialization, referred to as Xavier Uniform Initialization with formula limit = sqrt(6/(fan_in + fan_out)), designed to keep the scales of initialization approximately the same for all layers, used for sigmoid and tanh activations. He Uniform Initialization (also known as He at al. Initialization); formula: limit = sqrt(6/fan_in)), designed to work well with ReLU and its variants
                           ))
    model.add(layers.Dropout(rate=hp.Float('dropout_rate',min_value=0.2,max_value=0.5,step=0.1)))

    # Decoder
    model.add(layers.Dense(input_dim,activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2,sampling='LOG')
                                                  ),
                  loss='mse')
    return model

# Initializing the tuner
tuner = kt.BayesianOptimization(  # Using Bayesian Optimization for hyperparameter tuning
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='sparse_autoencoder_tuning'
)

# Setting up early stopping
early_stopping = keras.callbacks.EarlyStopping(  # Using early stopping to determine the number of epochs
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Running the Hyperparameter Search
tuner.search(X_train,X_train, epochs=100, validation_data=(X_test,X_test), batch_size=hp.Int('batch_size',min_value=32,max_value=128,steps=32), callbacks=[early_stopping],verbose=1)

# Getting the best Hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Best hyperparameters:
The optimal number of units in the encoder layer is
{best_hps.get('units')}.
The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
The optimal dropout rate is {best_hps.get('dropout')}.
The optimal batch size is {best_hps.get('batch_size')}.
The optimal kernel initializer is {best_hps.get('kernel_initializer')}.
""")
class SparseAutoencoder: # Class of Sparse Autoencoder model

    ''' Sparse autoencoder model with activation.

    Args:
        input_dim(int): An integer of the number of input dimensions
        encoding_dim(int): An integer of the number of encoded dimensions
        encoder(Sequential): a Sequential model of encoder
        decoder(Sequential): a Sequential model of decoder
    '''
    def __init__(self,input_dim, encoding_dim):
        super(SparseAutoencoder,self).__init__()
        # Initializing TPU
        self.trainable_variales = None
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
                Dense(units, activation='relu', input_shape=(input_dim,)), # Adding a dense (fully connected) layer as an encoding layer with 128 units and ReLU activation. The input shape is a vector length of input dimensions
                Dense(input_dim,activation='sigmoid')  # Adding a dense layer as a bottleneck layer with input dimensions as sigmoid as an activation function
            ])
            self.decoder = Sequential([
                Dense(units, activation='relu',input_shape=(encoding_dim,)),
                Dense(input_dim,activation='sigmoid')# Adding a dense layer as a bottleneck with input-dimension units and a sigmoid activation, used for reconstruction
            ])
            self.decoder = Sequential([  # Creating a sequential model
                Dense(units, activation='relu',input_shape=(encoding_dim,)),  # Adding a dense (fully connected) layer as an decoding layer with 128 units and ReLU activation. The input shape is a vector length of input dimensions
                Dense(input_dim,activation='sigmoid')
            ])
            self.autoencoder = Sequential([self.encoder,self.decoder])
            self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss='cross-entropy',metrics='accuracy',callbacks=[early_stopping]) # Specifying optimizer as Adam optimizer, the loss function as cross-entropy (used for multi-class classification), the metric that we are tracking is accuracy

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

# Initializing hyperparameters
units = best_hps.get('units')
learning_rate = best_hps.get('learning_rate')
dropout = best_hps.get('dropout')
kernel_initializer = best_hps.get('kernel_initializer')
batch_size = best_hps.get('batch_size')
num_epochs = 50

# Examples of usage
input_dim = 784
encoding_dim = 32

# Creating an instance of sparse autoencoder
sparse_autoencoder_model = SparseAutoencoder(input_dim,encoding_dim)

# Initializing kfold
fold = 1

# List to store loss values
train_losses = []

#List to store encoded inputs
all_encoded = []

# List to store weights
all_weights = []

start_time = t.time()

# Examples of usage
input_dim = 784
encoding_dim = 32

# Defining sparsity target and weight of the KL divergence term
sparsity_target = 0.05
beta = 0.1

for train_index, val_index in sgkf.split(X_train):
    print (f"Training fold: {fold}")
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    for epoch in range(num_epochs):
        for i in range(0, len(X_train_fold), batch_size):
            batch_train = X_train_fold[i:i+batch_size]
            batch_train = tf.convert_to_tensor(batch_train,dtype=tf.float32)
            batch_val = X_val_fold[i:i+batch_size]
            batch_val = tf.convert_to_tensor(batch_val,dtype=tf.float32)

            # Forward pass
            with tf.GradientTape() as tape:  # Recording operations for automatic differentiation using GradientTape with tape as context manager

                encoded, decoded = sparse_autoencoder_model(batch_train)  # Using the batch training data to perform a forward pass to obtain encoded inputs and decoded outputs
                recontruction_loss = tf.keras.losses.mean_squared_error(batch_train,decoded)  # Calculating the recontruction loss between the input and the decoded output using mean squared error (MSE). We use MSE for pixel-wise comparison, sensitivity for larger errors, simplicity, efficiency, common practice


            # Calculating the KL divergence
            p_hat = tf.reduce_mean(encoded,axis=0)  # Calculating the average activation, using tf.reduce_mean to compute the mean of elements across dimensions of a tensor
            kl_loss = tf.reduce_sum(kl_divergence(sparsity_target,p_hat))  # Calculating KL loss using sparsity_target and p_hat using tf.reduce_sum to compute the sum of elements across dimensions of a tensor
            loss = recontruction_loss + beta * kl_loss

            # Backward pass and optimization
            gradients = tape.gradient(loss,
                                      sparse_autoencoder_model.trainable_variables)  # Performing backward pass using loss and training variables (weights and biases in the model) to calculate gradients
            tensorflow.keras.layers.optimizer.apply_gradients(zip(gradients,
                                                                  sparse_autoencoder_model.trainable_variales))  # Applying computed gradients to trainable variables, putting gradients and trainable variables into a dictionary


            # Storing loss values in a list
            train_losses.append(loss.items())
            if (epoch + 1 ) % 10 == 0:  # Printing loss every 10 epochs
                print(f"Fold {fold} Epoch [{epoch+1}/{num_epochs}],Loss:{loss.item():.4f}")
        end_time = t.time()
        total_time = start_time - end_time

        # Extracting the encoded part for feature extraction
        encoded_features = encoded.numpy()  # Transforming the encoded representations into a Numpy array

        # Evaluating the model on the validation set
        val_decoded = sparse_autoencoder_model(batch_val)
        val_mse = tf.keras.losses.mean_squared_error(batch_val,val_decoded)  # Mean squared error loss
        val_mae = tf.keras.losses.mean_absolute_error(batch_val,val_decoded)  # Mean absolute error loss
        val_psnr = tf.image.psnr(batch_val,val_decoded,max_val=1.0)  # Peak signal-to-noise ratio, measuring the difference between the highest possible value of a signal to noise, the higher pnsr the better
        val_ssim = tf.image.ssim(batch_val,val_decoded,max_val=1.0)  # Structural similarity index, measuring the similarity between two images based on luminance, contrast and structure

        # Printing the metrics
        print(f" Validation MSE: {val_mse} "
              f"\n Validation MAE: {val_mae}"
              f"\n Validation PSNR: {val_psnr}"
              f"\n Validation SSIM: {val_ssim}")

        # Evaluating the model on the test set
        X_test_tensor = tf.convert_to_tensor(X_test)
        test_encoded = sparse_autoencoder_model(X_test_tensor)
        test_mse = tf.keras.losses.mean_squared_error(X_test_tensor,test_encoded)  # Mean squared error loss
        test_mae = tf.keras.losses.mean_absolute_error(X_test_tensor,test_encoded)  # Mean absolute error loss
        test_psnr = tf.image.psnr(X_test_tensor,test_encoded,max_val=1.0)  # Peak signal-to-noise ratio, measuring the difference between the highest possible value of a signal to noise, the higher pnsr the better
        test_ssim = tf.image.ssim(X_test_tensor,test_encoded,max_val=1.0)  # Structural similarity index, measuring the similarity between two images based on luminance, contrast and structure
        print(f" Test MSE: {test_mse} "
               f"\n Test MAE: {test_mae}"
               f"\n Test PSNR: {test_psnr}"
               f"\n Test SSIM: {test_ssim}")

# Visualizing the loss
plt.plot(train_losses,label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.show()

# Putting all encoded representations and weights into two lists
all_encoded.append(encoded.detach().numpy())
all_weights.append(sparse_autoencoder_model[0].weight.data.numpy())

# Visualizing the weights of the first layer
weights = sparse_autoencoder_model.encoder[0].weight.data.numpy()
plt.figure(figsize=(10,8))
sns.heatmap(weights,cmap="viridis",cbar=True)
plt.title('Heatmap of Weights - First Layer of Encoder')
plt.xlabel('Input Features')
plt.ylabel('Neurons')
plt.show()

# Visualizing encoded representations using PCA
encoded_np = np.concatenate(all_encoded,axis=0)
pca = PCA(n_components='mle')
encoded_pca = pca.fit_transform(encoded_np)
plt.scatter(encoded_pca[:,0],encoded_pca[:, 1])
plt.title("Encoded Representations (PCA)")
plt.show()

# Visualizing encoded representations using t-SNE
tsne = TSNE(n_components=2)
encoded_tSNE = tsne.fit_transform(encoded_np)
plt.scatter(encoded_tSNE[:,0],encoded_tSNE[:,1])
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Encoded Representations (t-SNE)")
plt.show()

class Mlp(nn.Module):
    '''Multi-layer Perceptron (MLP) module with optional dropout and activation.

    Args:
        encoded_features(int): An integer of the number of encoded features as input features.
        hidden_features(int): An integer of the number of hidden features.
        out_features(int): An integer of the number of output features.
        act_layer (nn.Module,optional): activation layer, Defaults to nn.GELU.
        drop(float,optional): Dropout rate. Defaults to 0.


    '''
    def __init__(self,encoded_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__()
        out_features = out_features or encoded_features
        hidden_features = hidden_features or encoded_features


#def process_model(use_data_parallel: bool = False, device_ids: list = None) -> torch.nn.Models:
