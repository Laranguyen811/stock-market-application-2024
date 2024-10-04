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
            for i in range(K):  # Iterating through K
                val = delta[t-1,i] + np.log(A[i,j])  # Assigning value to the addition between delta at time step t-1 and ith element and log of matrix A of state transition from i to j
                if val > max_val:
                    max_val = val  # Replacing max value with val every time val is greater than the current max value
                    max_state = i  # Replacing the max state with the ith state of the resulted max value
            delta[t,j] = max_val + np.log(B[j,X[t]])  # Assigning delta at time step t and state transition j to the addition between max value and the log of matrix B at state j and observation X at time step t
            psi[t,j] = max_state # Assigning matrix psi at time step t and state transition j to max state

    # Termination
    path = np.zeros(N, dtype=int)  # Creating an array of path filled with zeros with N dimension
    path[N-1] = np.argmax(delta[N-1])  # Assigning the N-1 th element of path to the output of the amximum function with input of the N-1 th element of delta

    # Backtracing
    for t in range(N-2, -1, -1):
        path[t] = psi[t+1,path[t+1]]  # Assigning path at time step t to psi at time step t+1 and path at time step t+1

    return path


def baum_welch_algorithm(X,A,B,pi, n_iter=100):
    ''' The Baum-Welch algorithm.
        Args:
            X(np.array): An np.array of sequence X of the observations
            A (np.array): An np.array of the transition matrix A transitioning from state Z_{i} to state Z_{j}
            B (np.array): An np.array of the probability distribution B of the observations
            alpha (np.array): An np.array of the probability distribution alpha of the observations up to the current time step (the filtered belief state)
            beta (np.array): An np.array of the conditional likelihood beta of the future evidence
    '''
    N = len(X)
    M = B.shape[1]  # Assigning M step to the number of rows of matrix B
    K = A.shape[0]  # Assigning K to the number of columns of matrix A


    for _ in range(n_iter):
        # E-step
        alpha = np.zeros((N,K))  # Filling alpha matrix with zeros with the dimensions of N and K
        beta = np.zeros((N,K))  # Filling beta matrix with zeros with the dimensions of N and K
        xi = np.zeros((N-1,K,K))  # Filling beta matrix with zeros with the dimensions of N-1,K and K
        gamma = np.zeros((N,K))  # Filling gamma matrix with zeros with the dimensions of N and K

        # Forward pass
        alpha[0] = pi * B[:, X[0]]  # Assigning the first element of alpha to the Hadamard product between pi and matrix B of the first observation of sequence X
        for t in range(1,N):  # In Baum-Welch algorithm the first step (t=0) is dealt with separately during initialisation. The loop then starts from t =1 and goes up to t=N-1, hence, we use range(1,N) to cover all the necessary steps.
            for i in range(K):
                beta[t,i] = np.sum(A[i,:] * B[:,X[t+1]] * beta[t+1])  # Assigning the matrix beta at time step t and state i to the sum of the Hadamard product between matrix A at state i and matrix B of matrix X of time step t + 1 and matrix beta at time step t + 1

        # Backward pass
        beta[-1] = 1
        for t in range(N-2, -1, -1):
            for j in range(K):
                alpha[t,j] = np.sum(alpha[t-1] * A[:,j]) *  B[j,X[t]]

        # Computing xi and gamma
        for t in range(N-1):
            denominator = np.sum(alpha[t] * beta[t])
            for i in range(K):
                gamma[t,i] = (alpha[t,i] * beta[t,i]) / denominator
                xi[t,i,:] = (alpha[t,i] * A[i,:] * B[:,X[t+1]] *beta[t+1]) / denominator

        gamma[-1] = alpha[-1] / np.sum(alpha[-1])  # Assigning the last elements of matrix gamma to the last elements of matrix alpha divided by the sum of the last elements of matrix alpha

        # M-step
        A = np.sum(xi,axis=0) / np.sum(gamma[:-1],axis=0)[:,None]  # Assigning matrix A to the sum of elements of the columns of matrix xi divided by the sum of the last elements of the columns of matrix gamma
        for j in range(M):
            mask = (X == j)  # Creating a boolean mask where in sequence X the observation is equal to j
            B[:,j] = np.sum(gamma[mask],axis=0) / np.sum(gamma,axis=0)
        pi = gamma[0]
    return A,B,pi








