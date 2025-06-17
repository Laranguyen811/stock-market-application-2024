import numpy as np
from typing import Tuple, Optional
from stock_market.hidden_markov_model import forward_algorithm, viterbi_algorithm_2, baum_welch_algorithm

def generate_synthetic_hmm_data(
    num_states: int,
    num_observations: int,
    sequence_length: int,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates synthetic data for a Hidden Markov Model.

    Args:
        num_states (int): The number of hidden states in the HMM.
        num_observations (int): The number of possible observable symbols.
        sequence_length (int): The length of the synthetic observation sequence to generate.
        random_seed (Optional[int]): Seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X (np.ndarray): The generated sequence of observations.
            - Z (np.ndarray): The true sequence of hidden states.
            - log_pi_true (np.ndarray): The true logged initial state probabilities.
            - log_A_true (np.ndarray): The true logged state transition matrix.
            - log_B_true (np.ndarray): The true logged emission matrix.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # 1. Define HMM Parameters (True Parameters)
    # Ensure probabilities sum to 1 before logging
    pi_true = np.random.dirichlet(np.ones(num_states))
    A_true = np.array([np.random.dirichlet(np.ones(num_states)) for _ in range(num_states)])
    B_true = np.array([np.random.dirichlet(np.ones(num_observations)) for _ in range(num_states)])

    # Log transform for numerical stability
    log_pi_true = np.log(pi_true)
    log_A_true = np.log(A_true)
    log_B_true = np.log(B_true)

    # 2. Generate Hidden States
    Z = np.zeros(sequence_length, dtype=int)
    X = np.zeros(sequence_length, dtype=int)

    # Initial state
    Z[0] = np.random.choice(num_states, p=pi_true)

    # Generate subsequent states and observations
    for t in range(sequence_length):
        if t > 0:
            Z[t] = np.random.choice(num_states, p=A_true[Z[t-1], :])
        X[t] = np.random.choice(num_observations, p=B_true[Z[t], :])

    return X, Z, log_pi_true, log_A_true, log_B_true

# Example usage:
if __name__ == "__main__":
    num_states = 3      # e.g., "sunny", "cloudy", "rainy"
    num_observations = 5 # e.g., "dry", "damp", "wet", "very_wet", "stormy"
    sequence_length = 50


    X_synthetic, Z_true, log_pi_true, log_A_true, log_B_true = generate_synthetic_hmm_data(
        num_states, num_observations, sequence_length, random_seed=42
    )

    print("--- Synthetic HMM Data Generated ---")
    print("\nGenerated Observations (X):")
    print(X_synthetic)
    print(f"Shape of X: {X_synthetic.shape}")

    print("\nTrue Hidden States (Z):")
    print(Z_true)
    print(f"Shape of Z: {Z_true.shape}")

    print("\nTrue Log Initial State Probabilities (log_pi_true):")
    print(log_pi_true)
    print(f"Shape of log_pi_true: {log_pi_true.shape}")

    print("\nTrue Log Transition Matrix (log_A_true):")
    print(log_A_true)
    print(f"Shape of log_A_true: {log_A_true.shape}")

    print("\nTrue Log Emission Matrix (log_B_true):")
    print(log_B_true)
    print(f"Shape of log_B_true: {log_B_true.shape}")

    print("\n--- Testing with your HMM functions ---")
    # You can now use these generated parameters to test your HMM functions.
    # For instance:
    log_alpha_test, likelihood_test = forward_algorithm(log_A_true, log_B_true[:, X_synthetic].T, log_pi_true)
    print(f"\nForward Algorithm Likelihood: {likelihood_test}")

    viterbi_path = viterbi_algorithm_2(X_synthetic, log_A_true, log_B_true, log_pi_true)
    print(f"Viterbi Path: {viterbi_path}")
    print(f"True Path:    {Z_true}")
    print(f"Viterbi path matches true path: {np.array_equal(viterbi_path, Z_true)}")

    # # Initialize random parameters for Baum-Welch training
    initial_log_pi = np.log(np.random.dirichlet(np.ones(num_states)))
    initial_log_A = np.log(np.array([np.random.dirichlet(np.ones(num_states)) for _ in range(num_states)]))
    initial_log_B = np.log(np.array([np.random.dirichlet(np.ones(num_observations)) for _ in range(num_states)]))
    print("\nStarting Baum-Welch training with random initial parameters...")
    learned_log_A, learned_log_B, learned_log_pi = baum_welch_algorithm(
         X_synthetic, initial_log_A, initial_log_B, initial_log_pi, n_iter=100
     )

    print("\nLearned Log Transition Matrix (log_A):")
    print(learned_log_A)
    print("\nTrue Log Transition Matrix (log_A_true):")
    print(log_A_true)

    print("\nLearned Log Emission Matrix (log_B):")
    print(learned_log_B)
    print("\nTrue Log Emission Matrix (log_B_true):")
    print(log_B_true)

    print("\nLearned Log Initial State Probabilities (log_pi):")
    print(learned_log_pi)
    print("\nTrue Log Initial State Probabilities (log_pi_true):")
    print(log_pi_true)