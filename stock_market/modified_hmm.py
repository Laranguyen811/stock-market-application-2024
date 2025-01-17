import numpy as np

import numpy as np

def log_normalize(log_u):
    """Normalize a log vector."""
    max_log_u = np.max(log_u)
    normalized = log_u - max_log_u - np.log(np.sum(np.exp(log_u - max_log_u)))
    return normalized

def forward_algorithm(log_A, log_B, log_pi):
    """Forward algorithm in log space."""
    N, K = log_B.shape
    log_alpha = np.zeros((N, K))

    # Initialize
    log_alpha[0] = log_normalize(log_B[0] + log_pi.reshape(-1))

    # Iterate
    for t in range(1, N):
        for j in range(K):
            log_alpha[t, j] = log_B[t, j] + log_normalize(log_alpha[t - 1] + log_A[:, j])

    log_likelihood = log_normalize(log_alpha[-1])
    return log_alpha, np.sum(log_likelihood)

def backward_algorithm(log_A, log_B):
    """Backward algorithm in log space."""
    N, K = log_B.shape
    log_beta = np.zeros((N, K))

    # Initialize
    log_beta[-1] = np.zeros(K)

    # Iterate
    for t in range(N - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = log_normalize(log_A[i] + log_B[t + 1] + log_beta[t + 1])

    return log_beta

def viterbi_algorithm(log_A, log_B, log_pi):
    """Viterbi algorithm in log space."""
    N, K = log_B.shape
    log_delta = np.zeros((N, K))
    psi = np.zeros((N, K), dtype=int)

    # Initialize
    log_delta[0] = log_B[0] + log_pi.reshape(-1)

    # Iterate
    for t in range(1, N):
        for j in range(K):
            log_delta[t, j] = np.max(log_delta[t - 1] + log_A[:, j]) + log_B[t, j]
            psi[t, j] = np.argmax(log_delta[t - 1] + log_A[:, j])

    # Termination
    path = np.zeros(N, dtype=int)
    path[-1] = np.argmax(log_delta[-1])

    # Backtracking
    for t in range(N - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path

def baum_welch_algorithm(X, log_A, log_B, log_pi, n_iter=100):
    """Baum-Welch algorithm in log space."""
    N = X.shape[0]
    K, M = log_B.shape

    for _ in range(n_iter):
        # E-step
        log_alpha, _ = forward_algorithm(log_A, log_B[:, X], log_pi)
        log_beta = backward_algorithm(log_A, log_B[:, X])

        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - np.logaddexp.reduce(log_gamma, axis=1)[:, np.newaxis]

        log_xi = np.zeros((N - 1, K, K))
        for t in range(N - 1):
            for i in range(K):
                for j in range(K):
                    log_xi[t, i, j] = log_alpha[t, i] + log_A[i, j] + log_B[j, X[t + 1]] + log_beta[t + 1, j]
            log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))

        # M-step
        log_A_num = np.logaddexp.reduce(log_xi, axis=0)
        log_A_denom = np.logaddexp.reduce(log_gamma[:-1], axis=0)
        log_A = log_A_num - log_A_denom[:, np.newaxis]

        for j in range(M):
            mask = (X == j)
            log_B[:, j] = np.logaddexp.reduce(log_gamma[mask], axis=0) - np.logaddexp.reduce(log_gamma, axis=0)

        log_pi = log_gamma[0]

    return np.exp(log_A), np.exp(log_B), np.exp(log_pi)

def train_and_evaluate_hmm(X, n_states, n_observations, n_iter=100, n_folds=5):
    """Train and evaluate HMM using K-fold cross-validation."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    log_likelihoods = []
    accuracies = []

    for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_index], X[test_index]

        # Initialize model parameters
        A = np.random.rand(n_states, n_states)
        A /= A.sum(axis=1, keepdims=True)
        B = np.random.rand(n_states, n_observations)
        B /= B.sum(axis=1, keepdims=True)
        pi = np.random.rand(n_states)
        pi /= pi.sum()

        log_A, log_B, log_pi = np.log(A), np.log(B), np.log(pi)

        # Training the model
        A, B, pi = baum_welch_algorithm(X_train, log_A, log_B, log_pi, n_iter)

        # Evaluating on test set
        _, log_likelihood = forward_algorithm(log_A, log_B[:, X_test], log_pi)
        log_likelihoods.append(log_likelihood)

        # Computing accuracy using Viterbi algorithm
        predicted_states = viterbi_algorithm(log_A, log_B[:, X_test], log_pi)
        true_states = np.array([np.argmax(B[:, obs]) for obs in X_test])  # Assuming the most likely state is the true state
        accuracy = np.mean(predicted_states == true_states)
        accuracies.append(accuracy)

        print(f"Fold {fold}:")
        print(f" Log-likelihood: {log_likelihood:.4f}")
        print(f" Accuracy: {accuracy:.4f}")

    return log_likelihoods, accuracies

