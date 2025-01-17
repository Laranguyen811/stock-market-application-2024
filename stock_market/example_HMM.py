import numpy as np


def forward(O, A, B, pi):
    N = A.shape[0]
    T = len(O)
    alpha = np.zeros((T, N))
    alpha[0, :] = pi * B[:, O[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1, :] * A[:, j]) * B[j, O[t]]
    return alpha


def backward(O, A, B):
    N = A.shape[0]
    T = len(O)
    beta = np.zeros((T, N))
    beta[T - 1, :] = 1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, O[t + 1]] * beta[t + 1, :])
    return beta


def baum_welch(O, N, M, max_iter=100):
    T = len(O)
    A = np.random.rand(N, N)
    A = A / A.sum(axis=1, keepdims=True)
    B = np.random.rand(N, M)
    B = B / B.sum(axis=1, keepdims=True)
    pi = np.random.rand(N)
    pi = pi / pi.sum()

    for _ in range(max_iter):
        alpha = forward(O, A, B, pi)
        beta = backward(O, A, B)

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            denom = np.sum(alpha[t, :] * np.sum(A * B[:, O[t + 1]] * beta[t + 1, :], axis=1))
            for i in range(N):
                numer = alpha[t, i] * A[i, :] * B[:, O[t + 1]] * beta[t + 1, :]
                xi[t, i, :] = numer / denom

        gamma = np.sum(xi, axis=2)
        pi = gamma[0, :]
        A = np.sum(xi, axis=0) / np.sum(gamma, axis=0, keepdims=True)
        gamma = np.vstack((gamma, np.sum(xi[T - 2, :, :], axis=0)))
        for k in range(M):
            B[:, k] = np.sum(gamma * (O == k).reshape(-1, 1), axis=0) / np.sum(gamma, axis=0)

    return A, B, pi


# Example usage
O = np.array([0, 1, 0, 2, 1, 0])  # Observations
N = 3  # Number of states
M = 3  # Number of observation symbols
A, B, pi = baum_welch(O, N, M)
print("Transition Probabilities:\n", A)
print("Emission Probabilities:\n", B)
print("Initial State Probabilities:\n", pi)

