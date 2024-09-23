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
from timm.models.layers import trunc_normal_  # a function to initialise weights of neural networks from a truncated normal distribution (bounding random variables either below, above or both, restricted to a specific range and preventing extreme values )


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
    '''Multi-layer Perceptron (MLP, involving a feedforward propagation function, a loss function and a backpropagation function) module with optional dropout and activation.

    Args:
        encoded_features(int): An integer of the number of encoded features as input features.
        hidden_features(int): An integer of the number of hidden features.
        out_features(int): An integer of the number of output features.
        act_layer (nn.Module,optional): activation layer, Defaults to nn.GELU (Gaussian Error Linear Unit, activation function (smooth and differentiable function across the real line, helping in gradient optimisation and mitigate the vanishing gradient problem) used in neural networks.
        drop(float,optional): Dropout rate. Defaults to 0.
    '''
    def __init__(self,encoded_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super().__init__()
        out_features = out_features or encoded_features
        hidden_features = hidden_features or encoded_features
        self.fc1 = nn.Linear(encoded_features,hidden_features)  # A fully-connected layer (where every neuron is connected to the previous neuron) using a linear function
        self.act = act_layer()  # An activation layer
        self.fc2 = nn.Linear(hidden_features,out_features)  # A fully-connected layer using a linear function
        self.drop = nn.Dropout(drop)  # A dropout layer is a regularisation technique to prevent overfitting during training, using randomness making the the model less sensitive to specific neurons and weight, promoting independence and robustness
        self.apply(self._init_weights)  # Applying initialised weights

        def _init_weights(self,m):
            if isinstance(m,nn.Linear):  # Check if initialised weights are only applied to an instance of nn.Linear (a fully-connected layer)
                trunc_normal_(m.weight,std=.02)  # Applying weights from a truncated normal distribution with standard deviation of 0.02 to ensure that initial weights are small and close to 0 so that we can achieve stable and efficient training (preventing exploding/vanishing gradients, efficiency, empirical success, normaization)
                if m.bias is not None:  # If we have assigned a value to bias
                    nn.init.constant_(m.bias,0)  # Initialising bias to 0 to ensure that all neurons have the same initial value of bias (uniformity, simplicity,symetry breaking,empirical success)
            elif isinstance(m,nn.LayerNorm):  # Check if initialised weights are applied to layer where Layer Normalisation is applied to a mini-batch of inputs (stabilising and accelerating the training of deep neural networks)
                nn.init.constant_(m.bias,0)  # Initialising bias to 0
                nn.init.constant_(m.weight,1.0)  #Initialising all weights to 1 to ensure that initial scaling does not affect the normalised values, ensuring the initial input is the same as the normalised input
        def forward(self,x):  # A feedforward propagation function
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


class GPSA(nn.Module):
    '''
    A gated positional self-attention (applying a gating mechanism to apply self-attention selectively, focusing on nearby regions of input) module.
    Args:
        dim (int): An integer of dimensionality of input.
        num_heads (int): An integer of a number of attention heads. Defaults to 8.
        qkv_bias (bool):  A boolean of whether to use bias in the linear layers of query, key, and value. Defaults to False.
        qk_scale (float): A float number of scaling factor for the query and key. Defaults to None.
        attn_drop (float): A float number of dropout rate for the attention mechanism. Defaults to 0.
        proj_drop (float): A dropout rate for the projection layer. Defaults to 0.
        locality_strength (float): A float number of the strength of the locality constraint. Defaults to 1.
        use_local_init (bool): A boolean of whether to use local initialisation. Defaults to True.
    '''
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: float = None, attn_drop: float = 0., proj_drop: float = 0.,locality_strength: float = 1.,use_local_init: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # Setting the scaling to scaling the dot product by head_dim ** -0.5 (helps in stablising the gradients during training (leading to more stable and efficient training, ensuring attention scores are consistent across layers and attention heads no matter the dimensionality of vectors)) if qk_scale is not provided

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Setting the query and key to a fully-connected layer of a linear function with dimension as inputs and dimension * 2 (computing both key and query vectors in 1 go for efficient training)  as outputs, bias is qkv_bias
        self.v = nn.Linear(dim,dim,bias=qkv_bias) # Setting the value to a fully-connected layer of a linear function with dimension as input and dimension as output and bias as qkv_bias

        self.attn_drop = nn.Dropout(attn_drop)  # Setting the drop rate of attention to a dropout layer
        self.proj = nn.Linear(dim, dim)  # Setting the projection layer (transforming input data to a different space, either higher or lower dimension) to a fully-connected layer (linear function) with input as dimensions and output as dimensions
        self.pos_proj = nn.Linear(3, num_heads)  # Setting the positional projection layer (setting positional data as input) to a fully-connected layer (linear function), taking a 3-positional dimension input and projecting it to a space of number-of-head dimension
        self.proj_drop = nn.Dropout(proj_drop)  # Setting a dropout rate for the projection layer to the dropout rate of a dropout layer
        self.locality_strength = locality_strength  # Setting a float number of the strength of the locality constraint
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))  # Setting the gating parameters to a tensor of 1 with dimensions of the number of heads
        self.apply(self._init_weights)  # Applying the initialised weights
        if use_local_init:  # If initialising the model parameters on each local machine or node independently
            self.local_init(locality_strength=locality_strength)  # use locality_strength for as a local initialisation

    def _init_weights(self,m):  # Initialising weights
        if isinstance(m,nn.Linear):  # If an object is an instance of nn.Linear
            nn.init.xavier_uniform_(m.weight)  # Using Xavier Uniform initialisation for more balance across different layers (effective training)
            if m.bias is not None:  # If bias is provided
                nn.init.constant_(m.bias,0)  # Setting a bias term to a constant value of 0


    def forward(self, x):  # Forward pass
        B, N, C = x.shape  # Setting B (batch size),N (the length of a sequence),C (embedding dimension) as the shape of x
        if not hasattr(self,'real_indices') or self.real_indices.size(1) != N:  # Check the condition if the real_indices attribute exists and the correct size before proceeding
            self.get_real_indices(N)  # If not, creating the real_indices tensor

        attn = self.get_attention(self)  # Getting an attention mechanism from the get_attention method
        v = self.v(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Reshaping the tensor v to dimension (B,N,2,the number of heads, the dimension for each attention head) and permutating for a specific order (2,0,3,1,4)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Performing matrix multiplication between attn and v, then transposing to swap the second and third dimensions of the tensor and reshaping it to dimension of (B, N, C)
        x = self.proj(x)  #  Setting x to the projection of x
        x = self.proj_drop(x)
        return x
    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # Reshaping the tensor qk to dimension (B,N,2,the number of heads, the dimension fo each attention head) and permutating for a specific order (2,0,3,1,4)
        q, k = qk[0], qk[1]  # Assigning values of q and k

        # Positional scores
        pos_score = self.real_indices.expand(B, -1, -1, -1)  # Ensuring the real_indices tensor is expanded to match the batch size B without changing other dimensions (aligning the positional information with batch size for further computations)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)  # Permutating pos_score after being positionally projected to a specific order (0,3,1,2)

        # Patch scores
        patch_score = (q @ k.transpose(-2, -1)) * self.scale  # Obtaining the scaled attention score by computing the matrix multiplication between query and key ( k with the last 2 dimensions transposed)
        patch_score = patch_score.softmax(dim=-1)  # Applying softmax function to the scaled attention score with the same dimension
        pos_score = pos_score.softmax(dim=-1)  # Applying the softmax function to positional information with the same dimension

        # Gating mechanism
        gating = self.gating_param.view(1, -1, 1, 1)  # Reshaping gating parameters without changing data with the specified dimension (1, -1, 1, 1) to align the dimension of gating parameters with other tensors to perform element-wise operations
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score  # Assigning attention mechanism to 1 - sigmoided gating tensor times scaled attention score, adding sigmoided gating tensor times positional score (gating score controlled by sigmoid function to demonstrate the importance of each score)
        attn /= attn.sum(dim=-1, keepdim=True)  # Normalising the attention score. Each element in the last dimension is the result of the sum of the corresponding element in the original attn tensor
        attn = self.attn_drop(attn)  # Applying the dropout rate to attn tensor
        return attn

    def get_attention_map(self, x, return_map=False):
        attn_map = self.get_attention(x).mean(0)  # Averaging attention head over batch
        # Calculating the distances

        distances = self.rel_indices.squeeze()[:, :, -1] ** 0.5  # Calculating the Eucledian distances by taking the squared root
        # Calculating the final distance
        dist = torch.einsum('nm,hnm->h',(distances,attn_map))  # Computing the weighted sum using attention map (using Einstein summation convention)
        dist /= distances.size(0)  # Normalising the results using the number of tokens
        # Calculating the distances is crucial to understand spatial relationships amongst the data (understanding how attention is distributed across spatial locations and how one part of input is affected by another）. Weighted distance measure helps us understand the influence of different parts of input based on the spatial relationships. It is very useful for image processing and natural language processing

        if return_map: # If there is return_map
            return dist, attn_map  # Return the distance and the attention map
        else:  # Otherwise
            return dist  # Return the distance

    def forward(self,x):  # Forward pass
        B, N, C = x.shape  # Extracting the shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)  # Reshaping the tensor q,k,v to dimension (B,N,3,the number of heads, the dimension for each attention head) and permutating for a specific order (2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # Obtaining the scaled attention heads by performing matrix multiplication between query and key (with the last and second to last dimensions transposed)
        attn = attn

    def local_init(self,locality_strength=1.):
        self.v.weight.data.copy_(torch.eye(self.dim))  # Initialising the values of weights to an identity matrix (a matrix that is product of the original matrix multiplied by the inverse matrix)
        locality_distance = 1  # Setting the locality distance

        kernel_size = int(self.num_heads ** 0.5)  # Calculating the kernel size based on square root of the number of heads
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2  # Determining the center of the kernel
        for h1 in range(kernel_size):  # Iterate over the kernel size
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2 # Calculate the position of the kernel
                self.pos_proj.weight.data[position, 2] = -1  # Setting the projection kernel weight for the third dimension
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance  # Setting the positional weight for the second dimension
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance  # Setting the positional weight for the first dimension
                self.pos_proj.weight.data *= locality_strength  # Scaling the positional project weights by the locality strength
    def get_real_indices(self, num_patches):
        img_size = int(num_patches ** 0.5)  # Calculating the image size based on the number of patches
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)  # Initialsing a tensor to store relative indices
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1,1)  # Computing the relative indices (positional encoding, capturing the relative positions in a sequence or grid) for the x and y dimensions
        indx = ind.repeat(img_size,img_size)  # Repeating the x indices to match the image size, representing the difference between x-coordinates of the patches
        indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)  # Repeating the y indices to match the image size, representing the difference in the y-coordinates of the patches
        indd = indx ** 2 + indy ** 2  # Computing the squared Euclidean distances between patches
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)  # Setting the indices for the 3rd dimension (the distance)
        rel_indices[:, :, :, 1] = indd.unsqueeze(0)  # Setting the indices for the 2nd dimension (indy)
        rel_indices[:, :, :, 0] = indd.unsqueeze(0)  # Setting the indices for the 1st dimension (indx)
        device = self.qk.weight.device  # Getting the device of the query-key weight
        self.rel_indices = rel_indices.to(device)  # Moving the relative indices to the same device as the query-key weights

class MHSA(nn.Module):
    ''' Multi-head Self-Attention (self-attention allows the model to weigh the importance of words in a sequence, multi-head performs multiple self-attention operations in parallel) module
        Args:
            dim(int): An integer of the number of dimensions.
            num_heads(int): An integer of a number of self-attention heads.
            qkv_bias (bool): A boolean of whether to use bias in the linear layers of query,key and vector or not. Defaults to False.
            qk_scale(float): A float number of a scaling factor for the query and key. Defaults to None.
            attn_drop (float): A drop rate for attention heads. Defaults to 0.
            proj_drop (float): A drop rate for the projection layer. Defaults to 0.
    '''
    def __init__(self,dim, num_heads=0,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.):
        super().__init__()  # A super class (parent class) dictating behaviors of child classes
        self.num_heads = num_heads  # A number of self-attention heads
        head_dim = dim/num_heads  # The number of dimensions per head
        self.scale = qk_scale or head_dim ** -0.5  # Setting the scaling factor to the scaling factor of the dot product between key and query (before being applied the softmax function) or the square root of number of dimensions per head if the key-query scale is not given.

        self.qkv = nn.Linear(dim,dim * 3,bias=qkv_bias)  # Setting query (information we want to find), key (relevance to the information we want to find) and value (actual value assigned to each word) vectors to a fully-connected layer (linear) with input as dimensions, output as 3 times dimensions (concatenating the query, key and value vectors into a single vector) and bias as qkv_bias
        self.attn_drop = nn.Dropout(attn_drop)  # Assigning the dropout rate of the attention head to a dropout layer
        self.proj = nn.Linear(dim,dim)  # Assigning the projection layer to a fully-connected layer with dimensions as inputs and dimensions as outputs
        self.proj_drop = nn.Dropout(proj_drop)  # Assigning the dropout rate of the projection layer to a dropout layer
        self.apply(self._init_weights)  # Applying the initialised weights

        def _init_weights(self,m):
            if isinstance(m,nn.Linear):  # If m is an instance of nn.Linear
                nn.init.xavier_uniform_(m.weight)  # Initialising the weight using Xavier (Glorot) Uniform Initialisation for balance of weights amongst layers, hence, efficient training
                if m.bias is not None:  # If bias is given
                    nn.init.constant_(m.bias, 0)  # Initialising bias to a constant of 0 for efficient training

        def get_attention_map(self,x,return_map=False):
            B, N, C = x.shape  # Extracting the shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Reshaping the tensor q,k,v to dimension (B,N,3,the number of heads, the dimension for each attention head) and permutating for a specific order (2,0,3,1,4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # Assigning query, key and value vectors
            attn_map = (q @ k.transpose(-2, -1)) * self.scale  # Obtaining the scaled attention map by performing matrix multiplication between query and key vectors with the last and the second to last dimensions transposed
            attn_map = attn_map.softmax(dim=-1).mean(0)  # Applying softmax function to the attention map with the same dimensions and calculating the mean along the columns to obtain the actual attention map

            # Calculating the distances
            img_size = int(N**.5)  # Assigning the image size as the integer of the square root of the number of batches
            ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)  # Obtaining the ind tensor by getting the 1-D tensor using the x and y dimensions of the image size
            indx = ind.repeat(img_size,img_size)  # Obtaining the indx tensor by repeating the image size as a row and the image soze as a column
            indy = ind.repeat_interleave(img_size,dim=0).repeat_interleave(img_size,dim=1)  # Obtaining the indx tensor by expanding the grid (repeating the image size for the image size times)
            indd = indx**2 + indy**2  # Obtaining indd tensor by adding the indx squared to indy squared
            distances = indd**.5  # Obtaining the Euclidian distances by squaring indd
            distances = distances.to('cuda')  # Putting distances to cuda

            dist = torch.einsum('nm,nhm->h',(distances,attn_map)) # Calculating the dist tensor using Einstein summation eith nm as the indices of the first tensor, nhm as the indices of the second tensor and h as the output indices. For each element of the output h, sum over the indices n and m by multiplying the corresponding distances and attention map (performing a weighted sum of attn_map tensor using the distances tensor as weights, and the result will the output tensor with h shape)
            dist /= N  # Dividing dist tensor with the number of batch to normalise
            if return_map:  # If there is a return map
                return attn_map,dist
            if None:  # If there is not a return map
                return dist

        def forward(self, x):  # Forward pass
            B, N, C = x.shape  # Extracting the shape of x
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2,-1)) * self.scale  # Reshaping the tensor q,k,v to dimension (B,N,3,the number of heads, the dimension for each attention head) and permutating for a specific order (2,0,3,1,4)
            attn = attn.softmax(dim=-1)  # Applying the softmax function to attention head with the same dimension
            attn = self.attn_drop(attn)  # Applying a dropout rate to attention heads

            x = (attn @ v).transpose(1,2).reshape(B, N, C)  # Obtaining the x tensor by performing matrix multiplication between attn and v tensors, transposing the first and the second dimensions, and reshaping them into the dimensions of (B, N, C)
            x = self.proj(x)  # Applying the projection layer to x
            x = self.proj_drop(x)  # Applying the dropout rate of the projection layer to x
            return x


class Block(nn.Module):
    '''
    A neural network block with optional GPSA or MHSA attention, normalisation, and MLP.
    '''
    def __init__(self, dim: int, num_heads:int,mlp_ratio:float = 4.,qkv_bias: bool = False, qk_scale: float = None, drop: float = 0., attn_drop: float = 0., drop_path: float = 0., act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm,use_gpsa: bool = True, **kwargs):
        '''
        Initialises the Block.
        Args:
            dim (int): An integer of the dimension of the input.
            num_heads (int): An integer of the number of heads.
            mlp_ratio (float): A float number of the ratio for MLP hidden dimension.
            qkv_bias (bool): A boolean of bias in QKV.
            qk_scale (float): A float number of scale for QK.
            drop (float): A float number of dropout rate.
            attn_drop (float): A float number of attention dropout rate.
            drop_path (float): A float number of drop path rate.
            act_layer (nn.Module): An activation layer.
            norm_layer (nn.Module): A normalisation layer.
            use_gpsa (bool): A boolean of using GPSA.
            **kwargs: Additional arguments.
        '''
        super().__init__()
        self.norm1 = norm_layer(dim)  # Applying a normalisation layer with dimensions as input
        self.use_gpsa = use_gpsa
        if self.use_gpsa:  # If using GPSA
            self.attn = GPSA(dim,num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,proj_drop=drop, **kwargs)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath is a regularization technique similar to dropout. nn.Identity() returns input as output
            self.norm2 = norm_layer(dim)  # Applying a normalisation layer with dimensions as input
            mlp_hidden_dim = int(dim * mlp_ratio)  # Obtaining the hidden dimensions for MLP as integer product between dimensions and the ratio for MLP hidden dimension (necessary for performance optimization, efficiency, and generalization capabilities)
            self.mlp = Mlp(encoded_features = dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                '''
                Forward pass of the Block.
                Args:
                    x (torch.Tensor): Input tensor.

                Returns:
                    torch.Tensor: Output tensor.
                '''
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x

class PatchEmbed(nn.Module):
    '''
        A module of image to patch embedding.

    Args:
        img_size (int): An integer of image size.
        patch_size (int): An integer of patch size.
    '''
    super().__init__()
    def __init__(self,img_size=224,patch_size=16, in_chans=3,embed_dim=768):
        ''' Args:
        img_size(int): An integer of image size.
        patch_size(int): An integer of patch size.
        in_chans (int): An integer of the number of input channels.
        embed_dim (int): An integer of the number of embedding dimensions.
        '''
        img_size = to_2tuple(img_size)  # Converting image size of a tuple of 2 elements
        patch_size = to_2tuple(patch_size)  # Converting patch size of a tuple of 2 elements
        num_patches = (img_size[1] // patch_size[1] * (img_size[0] // patch_size[0]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)  # Setting the projection layer to a 2D Convolutional Layer
        self.apply(self._init_weights)  # Applying initialised weights

    def forward(self,x):  # Forward pass
        B, C, H, W = x.shape  # Extracting the shape of x for the number of batches, the number of embedding dimensions, height and width respectively
        assert H == self.img_size[0] and W == self.img_size[1], \ f"Input image size ({H}*{W} doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1,2)
        return x

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):  # If m is an instance of nn.Linear
            nn.init.xavier_uniform_(m.weight)  # Initialising weights using Xavier Uniform Initialisation
            if isinstance(m,nn.Linear) and m.bias is not None:  # If m is an instance of nn.Linear and bias is given
                nn.init.constant_(m.bias,0)  # Initialising bias as a constant 0 for efficient training
            elif isinstance(m,nn.LayerNorm):  # If m is a constant of nn.LayerNorm
                nn.init.constant_(m.bias,0)  # Initialising bias as a constant 0 to keep the normalised layer the same
                nn.init.constant_(m.weight,1.0)  # Initialising weight as a constant 1 to keep the initial input the same as the normalised input


class HybridEmbed(nn.Module):
    '''
        CNN Feature Map Embedding as a module, from timm, used as a feature extractor.
        Args:
            backbone (nn.Module): CNN backbone to extract features.
            img_size (int or tuple): A int or tuple of input image size.
            feature_size (tuple,optional): A tuple of the size of the feature map.
            in_chans (int): A integer of the number of input channels.
            embed_dim (int): An integer of the dimension of the embedding.
    '''
    def __init__(self,backbone,img_size=224,feature_size=None,in_chans=3,embed_dim=768):
        super().__init__()
        assert isinstance(backbone,nn.Module), "backbone must be an instance of nn.Module"  # An assertion to ensure that backbone is an instance of nn.Module
        img_size = to_2tuple(img_size)  # Converting image size into a tuple of 2 elements
        self.img_size = img_size
        self.backbone = backbone

        if feature_size is None: # If feature size is not provided
            with torch.no_grad():  # Using torch with no gradient calculation
                training = backbone.training  # Training with backbone
                if training:  # If using training
                    backbone.eval()  # Using evaluation
                o = self.backbone(torch.zeros(1, in_chans,img_size[0], img_size[1]))[-1]  # Assigning the variable o to the last output of the backbone model after filling the input with zero-filled tensor with the specified shape.
                feature_size = o.shape[-2:]  # Assigning feature size to the tensor of o from the second to last column onwards.
                feature_dim = o.shape[1]  # Assigning feature dimension as the number of columns of tensor o
                backbone.train(training)  #Training backbone model

        else:
            feature_size = to_2tuple(feature_size)  # Converting the feature size to a tuple of 2 elements
            feature_dim = self.backbone.feature_info.channels(-1)  # Assigning the feature dimension to the second to last channel of the backbone model

            self.num_patches = feature_size[0]  * feature_size[1]  # Assigning the number of patches to the multiplication of the height and the width of the feature size
            self.proj = nn.Linear(feature_dim, embed_dim)  # Assigning the projection layer to the Linear layer with feature dimension as input and embedding dimension as output.
            self.apply(self._init_weights)  # Applying initialised weights

        def _init_weights(self, m):
            if isinstance(m,nn.Linear): # If m is an instance of nn.Linear
                nn.init.xavier_uniform_(m.weight)  # Initialising the weight using Xavier Uniform Initialisation for balance weights amongst layers and, hence, more efficient training
                if m.bias is not None:  # If bias is provided
                    nn.init.constant_(m.bias,0)  # Initialising bias as a constant of 0 for more efficient training


        def forward(self, x):  # Forward pass
            x = self.backbone(x)[-1]  # Assigning variable x to the last column of x as the output of the backbone model
            x = x.flatten(2).transpose(1,2)  # Flattening x to the second-level and transpose the second and the third columns
            x = self.proj(x)  #  Projecting x
            return x
#def process_model(use_data_parallel: bool = False, device_ids: list = None) -> torch.nn.Models:
