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
            np.array: A np.array of the normalised matrix gamma (the smoothed posterior, probability distribution of hidden states of the model given all the observations up to and include current time step and future observations) from 1 to N of the normalised product between matrix A and the dot product between the matrix psi at time t+1 and matrix beta at time t+1

    '''
    N = len(psi)
    beta = np.zeros(N,len(alpha[0]))  # beta is the conditional likelihood of the future evidence
    gamma = np.zeros(N, len(alpha[0]))

    # Initializing beta_N (beta of the Nth element of the sequence)
    beta[N - 1] = 1

    # Iterating through the sequence in reverse
    for t in range(N -2, -1, 1):
        beta[t], _ = normalize(A @ (psi[t+1] * beta[t+1]))  # Assigning beta at time t the normalised product between matrix A and the product between matrix psi at time t + 1 and matrix beta at t +1

    # Computing gamma
    for t in range(N):
        gamma[t], _ = normalize(alpha[t] * beta[t])

    return gamma

def viterbi_algorithm(X,A, B, pi):
    ''' A viterbi algorithm to compute the most probable sequence of hidden states, computing the shortest path through the trellis diagram of the Hidden Markov Model.

    Args:
        X (np.array): An np.array sequence X of the observations
        A (np.array): An np.array of the transition matrix A
        K (np.array): An np.array of matrix K of the transition matrices
        B (np.array): An np.array of the probability distribution B of the observations X_{t,k} given states Z_{t,j}
        pi (np.array): An np.array of the initial state distribution pi

    Returns:
        np.array: An np.array of the most probable sequence Z from 1 to N of hidden states
    '''
    N = len(X)
    K = A.shape[0] # Assigning K to the number of matrices A

    # Initializing delta and psi
    delta = np.zeros((N,K))
    psi = np.zeros((N,K), dtype=int)
    delta[0] = np.log(pi) + np.log(B[:, X[0]])

    # Iterating through A and B
    for t in range(1,N):
        for j in range(K):
            max_val = -np.inf  # Common practice for an algorithm finding a maximum value in a set of comparisons
            max_state = 0
            for i in range(K):
                val = delta[t-1,i] + np.log(A[i,j])
                if val > max_val:
                    max_val = val
                    max_state = i



