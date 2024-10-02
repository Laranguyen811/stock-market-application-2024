import hmmlearn
import numpy as np

# Hidden Markov Model (HMM) is a tool representing the probability of states given observed dat. The states are not directly observable, hence, hidden.
# State Z_{t} at time t depends on the previous state Z_{t-1} at time t-1. We call this first-order HMM

# The Forward-Backward Algorithm (a dynamic programming algorithm using message passing)

def normalize(u):
    ''' Normalize the vector input.
    Args:
        u(np.array): An np.array of input vector
    Returns:
        np.array: A normalized vector alpha
        int: a normalized constant
    '''
    C = np.sum(u)
    alpha = u/C
    return alpha, C

def forward(A, psi, pi):
    ''' Forward pass of the algorithm
    Args:
        A(np.array): An np.array of state transition matrix A
        psi(np.array): An np.array of the probability distribution of state Z_{t} given the observation Z_{1:T}
        pi(np.array): An np.array of the probability distribution of initial states Z_{1}
    Returns:
        np.array: A np.array of the sequence of normalised vector alpha from 1 to N
        float: A float of log probability of the observations X_{t} given the observation X_{1:t-1}
    '''
    N = len(psi)  # Assigning N to the length of psi
    alpha = np.zeros((N, len(pi)))  # Assigning alpha to a matrix filled with zeros with N as the number of columns and length of pi as a the number of rows
    C = np.zeros(N)  # Assigning C to a vector filled with zeros with the length of N

    # Initializing alpha_1 and C_1
    alpha[0], C[0] = normalize(psi[0] * pi)

    # Iterating through the sequence
    for t in range(1, N):
        alpha[t], C[t] = normalize(psi[t] * (A.T @ alpha[t-1]))  # Assigning alpha and C at time t to the normalized dot product between psi at time t time the transposed matrix A times matrix alpha at time t -1


    log_prob = np.sum(np.log(C))
    return alpha, log_prob

def backward_algorithm(A, psi, alpha):
    ''' A backward pass of the algorithm.
        Args:
            A(np.array): An np.array of state transition matrix A
            psi(np.array): An np.array of the probability distribution of state Z_{t} given the observation Z{1:T}
            alpha(np.array): An np.array of the normalised vector alpha from 1 to N

        Returns:
            np.array: A np.array of the normalised matrix gamma of the

    '''