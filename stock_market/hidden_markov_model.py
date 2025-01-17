from typing import Any, Tuple, List, Optional

import hmmlearn
import numpy as np
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit, NDArray
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, KFold
import warnings
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
    max_log_u = np.max(log_u)  # Preventing overflow when exponentiating u
    shifted_log_u = log_u - max_log_u  # Stabilising computation by shifting log values to prevent overflow and increase precision
    log_sum = np.log(np.sum(np.exp(shifted_log_u)))  # Computing the logarithm of the sum of the exponents in a numerically stable way
    normalized_log_u = log_u - max_log_u - log_sum  # Normalizing the log values so their exponentiated sum equals 1
    return normalized_log_u


def forward_algorithm(log_A, log_B, log_pi):
    ''' Forward pass of the algorithm
    Args:
        log_A(np.array): An np.array of logged state transition matrix A
        log_B(np.array): An np.array of the logged probability distribution of observation O at time t given the states Z_{1:T}
        log_pi(np.array): An np.array of the logged probability distribution of initial states Z_{1}
    Returns:
        np.array: A np.array of the sequence of normalised vector alpha from 1 to N
        float: A float of log probability of the observations X_{t} given the observation X_{1:t-1}
    '''
    N = len(log_B)
    K = log_A.shape[0]
    log_alpha = np.zeros((N,K))
    # Initializing log_alpha
    log_alpha[0] = log_B[0] + log_pi
    # Iterating through the sequence
    for t in range(1, N):
        for j in range(K):
            terms = log_alpha[t-1] + log_A[:, j]
            log_alpha[t,j] = log_sum_exp(terms) + log_B[t,j]  # Assigning log_alpha at time step t and state j to the Hadamard dot product between the log sum exponential of terms and log B at time step t and state j

    log_likelihood = log_sum_exp(log_alpha[-1])
    return log_alpha, log_likelihood

def backward_algorithm(log_A,log_B):
    ''' A backward pass of the algorithm.
        Args:
            A(np.array): An np.array of state transition matrix A
            log_B(np.array): An np.array of the probability distribution of the observation given a state
            alpha(np.array): An np.array of the normalised vector alpha from 1 to N

        Returns:
            np.array: A np.array of the normalised matrix log_beta (the smoothed posterior, probability distribution of hidden states of the model given all the observations up to and include current time step and future observations) from 1 to N of the normalised product between matrix A and the dot product between the matrix psi at time t+1 and matrix beta at time t+1

    '''
    N = len(log_B)
    K = log_A.shape[0]
    log_beta = np.zeros((N,K))  # beta is the conditional likelihood of the future evidence

    # Initializing log beta of the Nth element of the sequence
    log_beta[-1] = np.zeros(K)

    # Iterating through the sequence in reverse
    for t in range(N -2, -1, - 1):
        for i in range(K):
            terms = log_A[i,:] + log_B[t+1,:] + log_beta[t+1,:]
            log_beta[t,i] = log_sum_exp(terms)  # Assigning  log beta at time t and state i to the log sum exponential of the terms
    return log_beta

def viterbi_algorithm_1(X,log_A, log_B, log_pi):
    ''' A viterbi algorithm to compute the most probable sequence of hidden states, computing the shortest path through the trellis diagram of the Hidden Markov Model.

    Args:
        X (np.array): An np.array sequence X of the observations
        log_A (np.array): An np.array of the logged transition matrix A
        log_B (np.array): An np.array of the probability distribution B of the observations X_{t,k} given states Z_{t,j}
        log_pi (np.array): An np.array of the initial state distribution pi
        n_iter(int): A integer of the number of iterations
    Returns:
        np.array: An np.array of the most probable sequence Z from 1 to N of hidden states
    '''
    N = len(X)
    K = log_A.shape[0] # Assigning K to number of columns of matrix A

    # Initializing delta and psi
    delta: ndarray[Any, dtype[floating[_64Bit] | float_]] = np.zeros((N,K))  # delta is the probability as the combination of the transition from the previous state i at time t-1 and the most probable path leading to i
    psi = np.zeros((N,K), dtype=int)
    delta[0] = log_pi + log_B[:, X[0]]

    # Iterating through A and B
    for t in range(1,N):
        for i in range(K):
            for j in range(K):
                max_val = -np.inf  # Common practice for an algorithm finding a maximum value in a set of comparisons
                max_state = 0
                val = delta[t-1, j] + log_A[i, j]  # Assigning value to the addition between delta at time step t-1 and ith element and log of matrix A of state transition from i to j
                if val > max_val:
                    max_val = val  # Replacing max value with val every time val is greater than the current max value
                    max_state = i  # Replacing the max state with the ith state of the resulted max value
                    delta[t,j] = max_val + log_B[j,X[t]] # Assigning delta at time step t and state transition j to the addition between max value and the log of matrix B at state j and observation X at time step t
                    psi[t,j] = max_state  # Assigning matrix psi at time step t and state j to max state
            delta[t, j] = np.max(delta[t-1] + log_A[:,j] + log_B[j,X[t]])   # Assigning matrix delta at time step t and state j to the maximum value of the addition between matrix delta at time step t-1  and the log of matrix A at state j and the matrix B of observation X at time t
            psi[t,j] = np.argmax(delta[t-1] + log_A[:,j])  # Assigning the matrix psi at time t and state j to the indices of the matrix of the Hadamard dot product between matrix delta at time step t-1 and matrix log A at state j
    # Termination
    path = np.zeros(N, dtype=int)  # Creating an array of path filled with zeros with N dimension
    path[N-1] = np.argmax(delta[N-1])  # Assigning the N-1 th element of path to the output index of the maximum function with input of the N-1 th element of delta

    # Back tracing
    for t in range(N-2, -1, -1):
        path[t] = psi[t+1,path[t+1]]  # Assigning path at time step t to psi at time step t+1 and path at time step t+1

    return path


def viterbi_algorithm_2(X:NDArray, log_A: NDArray, log_B: NDArray, log_pi: NDArray) -> NDArray:
    ''' Viterbi algorithm to most probable sequence of hidden states, computing the shortest path through the trellis diagram of Hidden Markov Model.

    Args:
        X (np.ndarray): An np.ndarray sequence X of the observations.
        log_A (np.ndarray): An np.ndarray logged transition matrix A.
        log_pi (np.ndarray): An np.ndarray of the logged initial state distribution pi of the observations.
        log_B (np.ndarray): An np.ndarray of the logged probability distribution B of observation X_{t,k} given observation Z_{t,j}
    Returns:
        np.ndarray: An np.ndarray of the most probable sequence Z from 1 to N hidden states
    '''
    N = len(X)
    K = log_A.shape[0]

    # Initializing delta and psi
    log_delta = np.zeros((N,K))# delta is the probability as the combination of the transition from the previous state i at time t-1 and the most probable path leading to i
    psi = np.zeros((N,K), dtype=int)
    log_delta[0] = log_pi + log_B[:, X[0]].max(axis=1)
    for t in range(1,N):
        for j in range(K):
            temp = log_delta[t-1] + log_A[:,j]
            log_delta[t, j] = np.max(temp) + log_B[j, X[t]]  # Assigning matrix delta at time step t and state j to the maximum value of the addition between matrix delta at time step t-1  and the log of matrix A at state j and the matrix B of observation X at time t
            psi[t, j] = np.argmax(temp)  # Assigning matrix psi to the output of the maximum function for the addition between matrix delta at time t-1 and the log of matrix A at state j

    # Termination
    path = np.zeros(N, dtype=int)  # Filled the matrix path with zeros of N dimension
    path[N-1] = np.argmax(log_delta[N-1]) # Assigning the N-1 th element of path to the output of the maximum function with input of the N-1 th element of delta

    # Back tracing
    for t in range(N-2,-1,-1):
        path[t] = psi[t+1,path[t+1]]  # Assigning path at time step t to psi at time step t+1 and path at time step t+1
    return path[0] # Returning the most likely sequence

def log_sum_exp(log_probs: np.ndarray,axis: Optional[int] = None) -> np.ndarray:
    '''
    Takes log probabilities and returns the log transformation. We use this technique to minimise the effect of underflow or overflow when computing the log of a sum of exponentials.
    The first step is to find the maximum value, subtracting it from the all values (shifting the values to reduce the risk of overflow), find the exponential of all shifted values and add them all up. Then find
    the log of the summed exponentials and add the maximum value back to achieve the final result.
    Inputs:
        log_probs(float): A float number of log probabilities.
    Returns:
        float: A float number of log transformation
    '''
    max_log_prob = np.max(log_probs, axis=axis,keepdims=True)
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))
def baum_welch_algorithm (X:np.ndarray, log_A:np.ndarray, log_B:np.ndarray, log_pi: np.ndarray, n_iter: int =100, tol: float = 1e-6, min_prob: float = 1e-300) -> Tuple[ndarray[Any, dtype[Any]] | ndarray, ndarray, ndarray[Any, dtype[Any]] | ndarray]:

    ''' The Baum-Welch algorithm.
    Args:
        X(np.array): An np.array of sequence X of the observations
        log_A (np.array): An np.array of the logged transition matrix A transitioning from state Z_{i} to state Z_{j}
        log_B (np.array): An np.array of the logged probability distribution B of the observations
        log_pi (np.array): An np.array of the logged initial state distribution
        n_iter (int): Number of iterations for the algorithm
    Returns:
        log_A, log_B, log_pi: Updated model parameters
    '''
    validate_hmm_params(log_A, log_B, log_pi)
    N = len(X)
    M,K = log_B.shape # Assigning M step to the number of rows of matrix B and K to the number of columns of matrix A
    prev_likelihood = -np.inf
    best_likelihood = -np.inf
    best_params = (log_A.copy(),log_B.copy(),log_pi.copy())

    #Computing the probability for all observations
    B_obs = np.zeros((N, K))
    for iteration in range(n_iter):  # Using _ here since we don't need to assign a variable
        try:
            B_obs = log_B[:, X].T
            # E-step with numerical stability check
            log_alpha, current_likelihood = forward_algorithm(log_A, B_obs, log_pi)
            if not np.isfinite(current_likelihood):
                warnings.warn("Non-finite likelihood encountered. Using previous best parameters.")
                return best_params
            log_beta = backward_algorithm(log_A,B_obs)
            #Store best parameters
            if current_likelihood > best_likelihood:
                best_likelihood = current_likelihood
                best_params =(log_A.copy(),log_B.copy(),log_pi.copy())

            # Check convergence
            likelihood_improvement = current_likelihood - prev_likelihood
            if 0 <= likelihood_improvement < tol:
                print(f"Convergence after {iteration +1} iterations.")
                return best_params
            #Compute posterior probabilities
            log_gamma = compute_log_gamma(log_alpha,log_beta)
            # print(f"log gamma shape: {log_gamma.shape}")

            log_xi = compute_log_xi(log_alpha,log_beta,log_A,B_obs,X)

            # M-step with numerical stability safeguards
            log_A = update_transition_matrix(log_gamma, log_xi, min_prob)
            log_B = update_emission_matrix(log_gamma,X,M,K,min_prob)
            log_pi = log_gamma[0]

            # Normalise to prevent underflow/overflow
            log_A = normalise_log_matrix(log_A)
            log_B = normalise_log_matrix(log_B)
            log_pi = normalise_log_vector(log_pi)
            #print(f"log A shape: {log_A.shape},log B shape: {log_B.shape}, log pi shape: {log_pi.shape}")

            prev_likelihood = current_likelihood

        except (RuntimeWarning, FloatingPointError) as e:
            warnings.warn(f"Numerical Instability Encountered:{str(e)}. Using previous best parameters.")
            return best_params

        warnings.warn(f"Maximum iterations ({n_iter}) reached without convergence.")
        return best_params
def compute_log_gamma(log_alpha: np.ndarray, log_beta: np.ndarray) -> np.ndarray:
    """
    Compute log gamma values.
    Inputs:
        log_alpha(np.ndarray): An np.ndarray of logged matrix alpha of state Z from time 1 : N.
        log_beta(np.ndarray): An np.ndarray of logged matrix beta of observation X at time t given the states from t - 1 backwards.
    Returns:
        np.ndarray: An np.ndarray of logged matrix gamma values (smoothed posterior probabilities) of an observation given past and future evidence with improved numerical stability
    """
    log_gamma = log_alpha + log_beta
    if len(log_gamma.shape) == 2:
        new_log_gamma = log_gamma - log_sum_exp(log_gamma,axis=1)[: np.newaxis]  # Calculating the log sum exponential acros the axis 1, the states, then broadcasting to match the shapes for element-wise operations by adding a new dimension. Then perform the subtraction.
    elif len(log_gamma.shape) == 3:
        new_log_gamma = log_gamma - log_sum_exp(log_gamma, axis=2)[:, :, np.newaxis] # Calculating the log sum exponential acros the axis 2, a set of values like probabilities across states or categories, then broadcasting to match the shapes for element-wise operations by adding a new dimension. Then perform the subtraction.
    return new_log_gamma
def compute_log_xi(log_alpha: np.ndarray,log_beta: np.ndarray,log_A:np.ndarray, B_obs: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''
    Compute log xi (probabilities of transitioning from state i to state j given the model and the observations) values using vectorised operations if possible.
    Inputs:
        log_alpha(np.ndarray): An np.ndarray of logged matrix alpha of a state at time t given the previous observations.
        log_beta(np.ndarray): An np.ndarray of logged matrix beta of a state at time t given the future evidence.
        log_A(np.ndarray): An np.ndarray of logged transition matrix A.
        B_obs(np.ndarray): An np.ndarray of logged matrix of probability of an observation X at time t given a state at time t.
        X(np.ndarray): An np.ndarray of observations X.
    Returns:
        np.ndarray: An np.ndarray of logged xi values (probabilities of transitioning from state i to state j )
    '''
    N = len(X)
    K = log_A.shape[0]
    log_xi = np.zeros((N - 1, K, K))  # Filling beta matrix with zeros with the dimensions of N-1,K and K
    for t in range(N - 1):
        log_xi[t] = (log_alpha[t,:,np.newaxis] + log_A + B_obs[t+1] + log_beta[t + 1])  # Creating a new axis to broadcast matrices with different shapes, allowing for correct element-wise operations.

        log_xi[t] -= log_sum_exp(log_xi[t].reshape(-1))  # Reshaping log_xi at t to 1-D array
    return log_xi

def update_transition_matrix(log_gamma: np.ndarray, log_xi: np.ndarray, min_prob: float) -> np.ndarray:
    '''
    Update transition matrix with numerical stability checks.
    Inputs:
        log_gamma(np.ndarray): An np.ndarray of logged matrix alpha of a state at time t given the previous observations.
        log_xi(np.ndarray): An np.ndarray of logged values (probabilities of transitioning from state i to state j given the model and the observations)
        min_prob(float): A float number of minimum probability.
    Returns:
        np.ndarray: An np.ndarray of updated logged transition matrix A.
    '''
    # M-step
    K = log_gamma.shape[1]
    log_A_num = np.sum(log_xi, axis=0)
    if log_A_num.size == K * K:
        log_A_num_reshaped = log_A_num.reshape(K,K)
    else:
        raise ValueError(f" Cannot reshape array of size {log_A_num.size} into shape ({K} ,{K}")
    log_A_denominator = np.sum(log_gamma[:-1],axis=0)
    log_A = np.log(np.maximum(np.exp(log_A_num_reshaped) / np.maximum(np.exp(log_A_denominator)[:, np.newaxis], min_prob), min_prob))
    return log_A

def update_emission_matrix(log_gamma: np.ndarray, X: np.ndarray, M: int, K: int, min_prob: float) -> np.ndarray:
    '''
    Update emission matrix with vectorised operations and stability checks.
    Inputs:
        log_gamma(np.ndarray): An np.ndarray of logged gamma matrix of a state given past and future evidence.
        X(np.ndarray): An np.ndarray of observations X.
        M(np.ndarray): An np.ndarray of the row dimension of the logged transition matrix A.
        K(np.ndarray): An np.ndarray of the column dimension of the logged transition matrix A.
        min_prob(float): A float number of the minimum probability.
    Returns:
        np.ndarray: An np.ndarray of the emission matrix log B
    '''
    log_B = np.full((K,M), np.log(min_prob))
    for j in range(K):
        for k in range(M):
            mask = (X == k)
            if np.any(mask):
                log_B[j,k] = log_sum_exp(log_gamma[mask, j]) - log_sum_exp(log_gamma[:, j])

    return log_B

        #for j in range(K):
         #   for k in range(M):
         #       mask = (X == k)
          #      log_B[j,k] = log_sum_exp(log_gamma[mask,j]) - log_sum_exp(log_gamma[:,j])
        #log_pi = log_gamma[0]
    #else:
     #   print(f"Warning: Maximum iterations({n_iter}) reached without convergence")

   # return log_A,log_B,log_pi

def validate_hmm_params(log_A, log_B,log_pi):
    '''
    Validate HMM parameters.
    Args:
        log_A (np.array): an np.array of logged transition matrix A
        log_B (np.array): an np.array of logged probability distribution of an observation 0 at time t given a state Z_{1:T}
        log_pi (np.array): an np.array of logged initial state probability distribution
    Raises:
         ValueError: if the values are zero and of wrong input shapes and types
    '''
    if not isinstance(log_A,np.ndarray) or not isinstance(log_B,np.ndarray):
        raise TypeError("Transition and emission matrices must be numpy arrays")

    if log_A.shape[0] != log_A.shape[1]:
        raise ValueError("Transition matrix must be square")

    if log_A.shape[0] != log_B.shape[0]:
        raise ValueError("Inconsistent state dimensions between A and B matrices")

    if log_pi.shape[0] != log_A.shape[0]:
        raise ValueError("Initial distribution dimension doesn't match number of states")

def normalise_log_matrix(log_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize log probabilities in a matrix to prevent numerical instability.
    Inputs:
        log_matrix (np.ndarray): An np.ndarray of logged matrix.
    Returns:
        np.ndarray: An np.ndarray of normalised logged matrix.
    """
    return log_matrix - log_sum_exp(log_matrix, axis=1)[:, np.newaxis]

def normalise_log_vector(log_vector : np.ndarray) -> np.ndarray:
    '''
    Normalize log probabilities in a vector to prevent numerical issues
    Inputs:
        log_vector (np.ndarray): An np.ndarray of logged vector.
    Returns:
         np.ndarray: An np.ndarray of normalised logged vector.
    '''
    return log_vector - log_sum_exp(log_vector)