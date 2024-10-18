from typing import Any, Tuple, List

import hmmlearn
import numpy as np
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold
# Hidden Markov Model (HMM) is a tool representing the probability of states given observed data. The states are not directly observable, hence, hidden.
# State Z_{t} at time t depends on the previous state Z_{t-1} at time t-1. We call this first-order HMM.

# The Forward-Backward Algorithm (a dynamic programming algorithm using message passing)

def log_normalize(log_u):
    ''' Normalize the vector input.
    Args:
        log_u(np.array): An np.array of logged input vector
    Returns:
        np.array: A normalized vector alpha
        int: a normalized constant
    '''
    max_log_u = np.sum(log_u)
    if max_log_u == 0:
        max_log_u = np.finfo(float).eps
    alpha = log_u - max_log_u - np.log(np.sum(np.exp(log_u - max_log_u)))
    return alpha


def forward_algorithm(log_A, log_B, log_pi):
    ''' Forward pass of the algorithm
    Args:
        log_A(np.array): An np.array of logged state transition matrix A
        log_B(np.array): An np.array of the logged probability distribution of observation O at time t given the states Z_{1:T}
        pi(np.array): An np.array of the probability distribution of initial states Z_{1}
    Returns:
        np.array: A np.array of the sequence of normalised vector alpha from 1 to N
        float: A float of log probability of the observations X_{t} given the observation X_{1:t-1}
    '''
    N,K = log_B.shape # Assigning N,K to the dimensions of log_B with N equals the number of columns and K equal the number of rows
    log_alpha = np.zeros((N, K))  # Assigning alpha to a matrix filled with zeros with N as the number of columns and length of pi as a the number of rows

    # Initializing log_alpha
    log_alpha[0] = log_normalize(log_B[0] * log_pi)

    # Iterating through the sequence
    for t in range(1, N):
        log_alpha[t] = log_normalize(log_B[t] * (log_A.T @ log_alpha[t-1]))  # Assigning alpha and C at time t to the normalized dot product between psi at time t time the transposed matrix A times matrix alpha at time t -1

    log_probs = np.sum(log_normalize(log_alpha[-1]))
    return log_alpha, log_probs

def backward_algorithm(log_A,log_B):
    ''' A backward pass of the algorithm.
        Args:
            A(np.array): An np.array of state transition matrix A
            psi(np.array): An np.array of the probability distribution of state Z_{t} given the observation Z{1:T}
            alpha(np.array): An np.array of the normalised vector alpha from 1 to N

        Returns:
            np.array: A np.array of the normalised matrix gamma (the smoothed posterior, probability distribution of hidden states of the model given all the observations up to and include current time step and future observations) from 1 to N of the normalised product between matrix A and the dot product between the matrix psi at time t+1 and matrix beta at time t+1

    '''
    N,K = log_B.shape  # Assigning N,K to the dimensions of log_B with N equals the number of columns and K equal the number of rows
    beta = np.zeros((N,K))  # beta is the conditional likelihood of the future evidence
    gamma = np.zeros_like((N,K))

    # Initializing beta_N (beta of the Nth element of the sequence)
    beta[N-1] = 1

    # Iterating through the sequence in reverse
    for t in range(N -2, -1, - 1):
            beta[t], _ = log_normalize(A @ (psi[t+1] * beta[t+1]))  # Assigning beta at time t the normalised product between matrix A and the product between matrix psi at time t + 1 and matrix beta at t +1
    # Computing gamma
    for t in range(N):
        gamma[t], _ = log_normalize(alpha[t] * beta[t])

    return gamma

def viterbi_algorithm_1(X,A, B, pi):
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
    K = A.shape[0] # Assigning K to number of columns of matrix A

    # Initializing delta and psi
    delta: ndarray[Any, dtype[floating[_64Bit] | float_]] = np.zeros((N,K))  # delta is the probability as the combination of the transition from the previous state i at time t-1 and the most probable path leading to i
    psi = np.zeros((N,K), dtype=int)
    delta[0] = np.log(pi) + np.log(B[:, X[0]])

    # Iterating through A and B
    for t in range(1,N):
        for i in range(K):
            for j in range(K):
                max_val = -np.inf  # Common practice for an algorithm finding a maximum value in a set of comparisons
                max_state = 0
                val = delta[t-1, i] + np.log(A[i, j])  # Assigning value to the addition between delta at time step t-1 and ith element and log of matrix A of state transition from i to j
                if val > max_val:
                    max_val = val  # Replacing max value with val every time val is greater than the current max value
                    max_state = i  # Replacing the max state with the ith state of the resulted max value
                    delta[t,j] = max_val + np.log(B[j,X[t]])  # Assigning delta at time step t and state transition j to the addition between max value and the log of matrix B at state j and observation X at time step t
                    psi[t,j] = max_state  # Assigning matrix psi at time step t and state transition j to max state
            temp = delta[t-1] + np.log(A.T)
            delta[t, j] = np.max(delta[t-1] + np.log(A[:,j] + np.log(B[j,X[t]])))   # Assigning matrix delta at time step t and state j to the maximum value of the addition between matrix delta at time step t-1  and the log of matrix A at state j and the matrix B of observation X at time t
            psi[t,j] = np.argmax(delta[t-1] + np.log(A[:,j]))
    # Termination
    path = np.zeros(N, dtype=int)  # Creating an array of path filled with zeros with N dimension
    path[N-1] = np.argmax(delta[N-1])  # Assigning the N-1 th element of path to the output of the amximum function with input of the N-1 th element of delta

    # Back tracing
    for t in range(N-2, -1, -1):
        path[t] = psi[t+1,path[t+1]]  # Assigning path at time step t to psi at time step t+1 and path at time step t+1

    return path


def viterbi_algorithm_2(X, A, B, pi):
    ''' Viterbi algorithm to most probable sequence of hidden states, computing the shortest path through the trellis diagram of Hidden Markov Model.

    Args:
        X (np.ndarray): An np.ndarray sequence X of the observations.
        A (np.ndarray): An np.ndarray transition matrix A.
        pi (np.ndarray): An np.ndarray of the initial state distribution pi of the observations.
        B (np.ndarray): An np.ndarray of the probability distribution B of observation X_{t,k} given observation Z_{t,j}
    Returns:
        np.ndarray: An np.ndarray of the most probable sequence Z from 1 to N hidden states
    '''
    N,K = len(X), A.shape[0]  # Assigning N to the length of X and K to the number of matrices A

    # Initializing delta and psi
    delta: ndarray[Any, dtype[floating[_64Bit] | float_]] = np.zeros((N,K)) # delta is the probability as the combination of the transition from the previous state i at time t-1 and the most probable path leading to i
    psi = np.array((N,K), dtype=int)
    delta[0] = np.log(pi) + np.log(B[:,X[0]])

    for t in range(1,N):
        for j in range(K):
            temp = delta[t-1] + np.log(A[:,j])
            delta[t,j] = np.max(temp) + np.log(B[j,X[t]])  # Assigning matrix delta at time step t and state j to the maximum value of the addition between matrix delta at time step t-1  and the log of matrix A at state j and the matrix B of observation X at time t
            psi[t,j] = np.argmax(temp)  # Assigning matrix psi to the output of the maximum function for the addition between matrix delta at time t-1 and the log of matrix A at state j

    # Termination
    path = np.zeros(N, dtype=int)  # Filled the matrix path with zeros of N dimension
    path[N-1] = np.argmax(delta[N-1]) # Assigning the N-1 th element of path to the output of the amximum function with input of the N-1 th element of delta

    # Back tracing
    for t in range(N-2,-1,-1):
        path[N-1] = psi[t+1,path[t+1]]  # Assigning path at time step t to psi at time step t+1 and path at time step t+1

    return path
def log_sum_exp(log_probs):
    '''
    Takes log probabilities and returns the log transformation
    Inputs:
        log_probs(float): A float number of log probabilities.
    Returns:
        float: A float number of log transformation
    '''
    max_log_prob = np.max(log_probs)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))
def baum_welch_algorithm(X, A, B, pi, n_iter=100):
    ''' The Baum-Welch algorithm.
    Args:
        X(np.array): An np.array of sequence X of the observations
        A (np.array): An np.array of the transition matrix A transitioning from state Z_{i} to state Z_{j}
        B (np.array): An np.array of the probability distribution B of the observations
        pi (np.array): An np.array of the initial state distribution
        n_iter (int): Number of iterations for the algorithm
    Returns:
        A, B, pi: Updated model parameters
    '''
    N, M, K = len(X), B.shape[1], A.shape[0]  # Assigning M step to the number of rows of matrix B and K to the number of columns of matrix A

    for _ in range(n_iter):
        # E-step
        log_alpha = np.zeros((N, K))  # Filling alpha matrix with zeros with the dimensions of N and K
        log_beta = np.zeros((N, K))  # Filling beta matrix with zeros with the dimensions of N and K
        xi = np.zeros((N-1, K, K))  # Filling beta matrix with zeros with the dimensions of N-1,K and K
        gamma = np.zeros((N, K))  # Filling gamma matrix with zeros with the dimensions of N and K

        # Forward pass
        log_alpha[0] = np.log(pi) * np.log(B[:, X[0]])  # Assigning the first element of alpha to the Hadamard product between pi and matrix B of the first observation of sequence X
        for t in range(1, N):  # In Baum-Welch algorithm the first step (t=0) is dealt with separately during initialisation. The loop then starts from t =1 and goes up to t=N-1, hence, we use range(1,N) to cover all the necessary steps.
            log_alpha[t] = log_sum_exp(log_alpha[t-1] + np.log(A)) * np.log(B[:, X[t]])  # Assigning the matrix beta at time step t and state i to the sum of the Hadamard product between matrix A at state i and matrix B of matrix X of time step t + 1 and matrix beta at time step t + 1

        # Backward pass
        log_beta[-1] = 0
        for t in range(N-2, -1, -1):
            log_beta[t] = log_sum_exp(np.log(A) + np.log(B[:, X[t+1]]) * log_beta[t+1])  # Assigning matrix beta at time step t and state j to the sum of matrix A at state j and matrix B at observation of time step t +1 and matrix beta at time step t + 1

        # Computing xi and gamma
        for t in range(N-1):
            log_denominator = log_sum_exp(log_alpha[t] + log_beta[t])
            if log_denominator == 0:
                log_denominator = np.finfo(float).eps # use a small non-zero value instead of 0
            gamma[t] = np.exp(log_alpha[t] + log_beta[t] - log_denominator)
            xi[t] = np.exp(log_alpha[t] + np.log(A) + np.log(B[:,X[t+1]]) + log_beta[t+1] - log_denominator)  # Assigning matrix xi at time step t to the outer product of matrix alpha at time step t and state i and the Hadamard product between matrix B of observation X at time step t + 1, multiplied with matrix A divided by the denominator

        gamma[-1] = np.exp(log_alpha[-1] - log_sum_exp(log_alpha[-1]))  # Assigning the last elements of matrix gamma to the last elements of matrix alpha divided by the sum of the last elements of matrix alpha

        # M-step
       # A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, None]  # Assigning A to the sum of matrix x along the rows divided by the sum of matrix gamma till the last column along the rows, and turning this into a column vector
    #    B[:, j] = np.array([np.sum(gamma[:, X == j], axis=0) for j in range(M)]).T  # Assigning B to the transposed array of the column of the sum of matrix gamma where observation X is at state j
     #   B /= np.sum(gamma, axis=0)[:, None]  # Dividing matrix B by the column vector sum of matrix gamma
      #  pi = gamma[0]

        A = np.sum(xi, axis=0) / np.sum(gamma[:-1, :, np.newaxis], axis=0)
        for j in range(M):
            B[:, j] = np.sum(gamma[X == j], axis=0)
        B /= np.sum(gamma, axis=0)[:, np.newaxis]
        pi = gamma[0]
    return A, B, pi

def train_and_evaluate_hmm(X: np.ndarray, n_states: int, n_observations: int,n_iter: int = 100, n_folds: int =5) -> Tuple[List[float], List[float]]:
    '''
    Train and evaluate HMM using K-fold cross-validation.

    Args:
        X (np.ndarray): The full dataset of observations
        n_states (int): The integer of the number of hidden states.
        n_observations (int): The integer of the number of observations.
        n_iter (int): The integer of the number of iterations for Baum-Welch algorithm.
        n_folds (int): The integer of the number of folds for cross-validation.

    Returns:
        Tuple[List[float], List[float]]: List of log-likelihoods and accuracies for each fold.

    '''
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    log_likelihoods = []
    accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(X),1):
        X_train, X_test = X[train_index], X[test_index]

        # Initializing model parameters
        A = np.random.rand(n_states, n_states)
        A /= A.sum(axis=1, keepdims=True)
        B = np.random.rand(n_states, n_observations)
        B /= B.sum(axis=1, keepdims=True)
        pi = np.random.rand(n_states)
        pi /= pi.sum()

        # Training the model
        A, B, pi = baum_welch_algorithm(X_train, A, B, pi, n_iter)

        # Evaluating on test set
        alpha, log_likelihood = forward_algorithm(A, B[:,X_test], pi)
        log_likelihoods.append(log_likelihood)

        # Computing accuracy using Viterbi algorithm
        predicted_states = viterbi_algorithm_2(X_test, A, B, pi)
        true_states = np.array([np.argmax(B[:, obs]) for obs in X_test])  # Assuming the most likely state is the true state
        accuracy = np.mean(predicted_states == true_states)
        accuracies.append(accuracy)

        print(f" Log-likelihood: {log_likelihood:.4f}")
        print(f" Accuracy: {accuracy:.4f}")

    return log_likelihoods, accuracies









