import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Any,Optional
import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from pandas import DataFrame
from sklearn.model_selection import KFold
from stock_market.hidden_markov_model import log_normalize,forward_algorithm,backward_algorithm,viterbi_algorithm_2,baum_welch_algorithm, compute_log_gamma,compute_log_xi,log_sum_exp,normalise_log_matrix,normalise_log_vector,update_emission_matrix,update_transition_matrix,validate_hmm_params

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd


def prepare_sequences(data: np.ndarray, sequence_length: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for training, adapted for small datasets.

    Args:
        data: Input array of stock movements
        sequence_length: Length of each sequence (flexible)

    Returns:
        Tuple of sequences and their corresponding labels
    """
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError('Data should be discretized before preparing sequences')
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length):
        sequence = data[i:(i + sequence_length)]
        label = data[i + sequence_length]
        sequences.append(sequence)
        labels.append(label)

    return np.array(sequences), np.array(labels)

def discretise_data(data: np.ndarray, n_bins: int =5) -> np.ndarray:
    '''
    Discretize continuous data into bins.
    Inputs:
        data: Continuous data array.
        n_bins: number of discrete states
    Returns:
        list: a list of discretised data
    '''
    return np.digitize(data, bins=np.quantile(data,np.linspace(0,1,n_bins+1)[1:-1])) -1
def initialize_hmm_parameters(n_states: int, n_observations: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize HMM parameters with proper dimensions."""
    # Add small epsilon to prevent log(0)
    epsilon = 1e-5

    # Initialize transition matrix
    A = np.random.rand(n_states, n_states) + epsilon
    A = A / A.sum(axis=1, keepdims=True)
    log_A = np.log(A)
    # Initialize emission matrix
    B = np.random.rand(n_states, n_observations) + epsilon
    B = B / B.sum(axis=1, keepdims=True)
    log_B = np.log(B)

    # Initialize initial state distribution
    pi = np.ones(n_states) / n_states
    log_pi = np.log(pi)


    return log_A, log_B, log_pi


def generate_realistic_stock_data(
        n_days: int = 500,
        initial_price: float = 100.0,
        volatility_states: List[float] = None,
        trend_states: List[float] = None,
        state_transition_probs: Optional[List[List[float]]] = None,
        mean_reversion_strength: float = 0.1,
        price_change_threshold: float = 0.001,
        random_seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate realistic stock market data with multiple market regimes.

    Args:
        n_days: Number of trading days to simulate
        initial_price: Starting price of the stock
        volatility_states: List of volatilities for different market regimes
                         (default: [0.01, 0.02, 0.03] for low/med/high)
        trend_states: List of trend values for different market regimes
                     (default: [-0.002, 0.001, 0.003] for down/flat/up)
        state_transition_probs: Matrix of state transition probabilities
        mean_reversion_strength: Strength of mean reversion (0 to 1)
        price_change_threshold: Threshold for categorizing price changes
        random_seed: Random seed for reproducibility

    Returns:
        Tuple containing:
        - price_changes: Array of discretized price movements (0:down, 1:flat, 2:up)
        - market_states: Array of market regime states
        - df: DataFrame with full price and return data
    """
    # Set default parameters if not provided
    if volatility_states is None:
        volatility_states = [0.01, 0.02, 0.03]  # Low, Medium, High volatility

    if trend_states is None:
        trend_states = [-0.002, 0.001, 0.003]  # Bearish, Neutral, Bullish

    if state_transition_probs is None:
        # Default transition matrix (tendency to stay in current state)
        state_transition_probs = [
            [0.95, 0.03, 0.02],  # Low volatility state transitions
            [0.05, 0.90, 0.05],  # Medium volatility state transitions
            [0.02, 0.03, 0.95]  # High volatility state transitions
        ]

    # Validate inputs
    assert len(volatility_states) == len(trend_states), "Volatility and trend states must have same length"
    assert all(len(row) == len(volatility_states) for row in state_transition_probs), "Invalid transition matrix"
    assert all(abs(sum(row) - 1.0) < 1e-10 for row in state_transition_probs), "Transition probabilities must sum to 1"

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize arrays
    prices = np.zeros(n_days)
    prices[0] = initial_price
    returns = np.zeros(n_days)
    market_states = np.zeros(n_days, dtype=int)
    price_changes = np.zeros(n_days, dtype=int)

    # Generate initial state randomly
    current_state = np.random.choice(len(volatility_states))
    market_states[0] = current_state

    # Generate time series
    for t in range(1, n_days):
        # Transition to next state based on transition matrix
        current_state = np.random.choice(
            len(volatility_states),
            p=state_transition_probs[current_state]
        )
        market_states[t] = current_state

        # Get current state parameters
        current_volatility = volatility_states[current_state]
        current_trend = trend_states[current_state]

        # Calculate mean reversion component
        price_deviation = (prices[t - 1] - initial_price) / initial_price
        mean_reversion = -mean_reversion_strength * price_deviation

        # Generate daily return
        daily_return = (
                current_trend +  # Trend component
                mean_reversion +  # Mean reversion
                np.random.normal(0, current_volatility)  # Random noise
        )
        returns[t] = daily_return

        # Update price
        prices[t] = prices[t - 1] * (1 + daily_return)

        # Categorize price change
        if daily_return < -price_change_threshold:
            price_changes[t] = 0  # Down
        elif daily_return > price_change_threshold:
            price_changes[t] = 2  # Up
        else:
            price_changes[t] = 1  # Flat

    # Create DataFrame with results
    dates = pd.date_range(
        start=datetime.now(),
        periods=n_days,
        freq='B'  # Business days
    )

    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Return': returns,
        'Market_State': market_states,
        'Price_Change': price_changes,
        'Volatility': [volatility_states[s] for s in market_states],
        'Trend': [trend_states[s] for s in market_states]
    })

    return price_changes, market_states, df
def generate_data_and_train_hmm(
    n_days: int = 500,
    n_bins: int = 5,
    initial_price: float = 100.0,
    random_seed: int = 42,
    sequence_length: int = 3,
    n_folds = 5,
    n_states: int = 3,
    n_observations: int =3
) -> tuple[list[Any], list[ndarray[Any, dtype[Any]]], ndarray[Any, dtype[Any]], list[Any], DataFrame]:
    """
    Generate realistic stock market data with multiple market regimes.

    Args:
        n_days: Number of trading days to simulate
        initial_price: Starting price of the stock
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (price_changes, market_states, price_data_df)
    """
    global predicted_states
    price_changes, market_states, price_data_df = generate_realistic_stock_data(n_days=n_days, random_seed=random_seed)

    #Discretize the returns data
    discrete_returns = discretise_data(price_data_df['Return'].values,n_bins=n_bins)
    # Prepared sequences
    X,y = prepare_sequences(discrete_returns,sequence_length=sequence_length)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold = 1
    log_likelihoods: list[Any] = []
    accuracies = []
    predicted_states = []
    for fold,(train_index,test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Initializing model parameters
        log_A,log_B,log_pi = initialize_hmm_parameters(n_states,n_observations)
        # Validate parameters
        validate_hmm_params(log_A, log_B, log_pi)
        # Train the model
        try:
            for train_seq in X_train:
                log_A,log_B,log_pi = baum_welch_algorithm(train_seq,log_A,log_B,log_pi,n_iter=100)
            # Evaluate on test set
            for seq in X_test:
                B_obs = np.zeros((len(seq),n_states))
                for t in range(len(seq)):
                    B_obs[t] = log_B[:,seq[t]]
                alpha, log_likelihood = forward_algorithm(log_A,B_obs,log_pi)
                log_likelihoods.append(log_likelihood)
            # Compute using Viterbi algorithm
            predicted_state = viterbi_algorithm_2(X_test,log_A,log_B,log_pi)
            true_states = np.array(np.max(log_B[:,obs]) for obs in X_test)
            accuracy = np.mean(predicted_states == true_states)
            accuracies.append(accuracy)
            predicted_states.append(predicted_state)
        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            continue

    # Adding predictions to DataFrame
    results_df = price_data_df.copy()
    results_df['Predicted_State'] = np.nan  # Initializing with NaN

    return log_likelihoods,predicted_states,true_states,accuracies,results_df

def evaluate_hmm(results_df: pd.DataFrame) -> dict:
    """
    Calculate performance metrics for the predictions.
    Inputs:
        results_df(DataFrame): A DataFrame containing the results
    Returns:
        dict: A dictionary of metrics
    """
    mask = ~results_df['Predicted_State'].isna()  # the '~' operator inverts these boolean values
    actual_states = results_df.loc[mask,'Market_State']
    predicted_states = results_df.loc[mask,'Predicted_State']

    # Calculating accuracy
    accuracy = np.mean(actual_states == predicted_states)

    # Calculating state-wise accuracy
    state_accuracy = {}
    for state in range(3):
        state_mask = actual_states = state
        if np.any(state_mask):
            state_accuracy[f'State_{state}_accuracy'] = np.mean(predicted_states[state_mask] == actual_states[state_mask])
    metrics = {'overall_accuracy': accuracy,
               **state_accuracy}
    return metrics
def plot_market_simulation(price_data_df: pd.DataFrame) -> None:
    """Plot the simulated market data with state annotations."""
    plt.figure(figsize=(15, 10))

    # Plot 1: Price Series with Market States
    plt.subplot(2, 1, 1)
    plt.plot(price_data_df.index, price_data_df['Price'], label='Stock Price', color='blue', alpha=0.7)

    # Color background based on market state
    for state in range(3):
        state_mask = price_data_df['Market_State'] == state
        plt.fill_between(price_data_df.index, price_data_df['Price'].min(), price_data_df['Price'].max(),
                         where=state_mask, alpha=0.2,
                         label=f'State {state}')

    plt.title('Simulated Stock Price with Market States')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Returns Distribution
    plt.subplot(2, 1, 2)
    plt.hist(price_data_df['Return'], bins=50, density=True, alpha=0.7)
    plt.title('Distribution of Returns')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# def train_and_evaluate_hmm(price_changes: np.ndarray = price_changes, n_states: int = 3, sequence_length: int = 5, n_observations: int =3,n_iter: int = 100, n_folds: int = 5) -> Tuple[List[float], List[float]]:
  #  '''
   # Train and evaluate HMM using K-fold cross-validation.

   # Args:
    #    X (np.ndarray): The full dataset of observations
     #   n_states (int): The integer of the number of hidden states.
      #  n_observations (int): The integer of the number of observations.
      #  n_iter (int): The integer of the number of iterations for Baum-Welch algorithm.

    # Returns:
      #  Tuple[List[float], List[float]]: List of log-likelihoods and accuracies for each fold.

    #'''
    # kf = KFold(n_splits= n_folds, shuffle=True, random_state=42)
   # log_likelihoods = []
   # accuracies = []
    # X,y = prepare_sequences(price_changes, sequence_length)
    # for fold, (train_index, test_index) in enumerate(kf.split(X),1):
      #  X_train, X_test = X[train_index], X[test_index]
      #  y_train, y_test = y[train_index], y[test_index]

        # Initializing model parameters
      #  log_A,log_B,log_pi = initialize_hmm_parameters(n_states, n_observations)

        # Validating parameters
       # validate_hmm_params(log_A,log_B, log_pi)
        # Training the model
         # try:
           # for train_seq in X_train:
            #    log_A,log_B,log_pi = baum_welch_algorithm(train_seq, log_A, log_B, log_pi, n_iter)

            # Evaluating on test set
            # for seq in X_test:
              #  B_obs = np.zeros((len(seq),n_states))
              #  for t in range(len(seq)):
               #     B_obs[t] = log_B[:,seq[t]]
                    #print(f"log B obs shape {B_obs.shape}")
               # alpha, log_likelihood = forward_algorithm(log_A, B_obs, log_pi)
            #print(f"log A shape {log_A.shape}, B obs shape {B_obs.shape}, log pi shape {log_pi.shape}")
            #print(f"Alpha:{alpha}, log_likelihood: {log_likelihood}")
                # log_likelihoods.append(log_likelihood)

            # Computing accuracy using Viterbi algorithm
            # predicted_states = viterbi_algorithm_2(X_test, log_A, log_B, log_pi)
            # print(f"X_test: {X_test.shape},log A shape {log_A.shape},log B shape {log_B.shape},log pi shape {log_pi.shape}")
            # true_states = np.array([np.argmax(log_B[:, obs]) for obs in X_test])  # Assuming the most likely state is the true state
            # accuracy = np.mean(predicted_states == true_states)
            # accuracies.append(accuracy)
        # except Exception as e:
          #  print(f"Errors in fold {fold}: {str(e)}")
          #  continue
        #print(f"Log-likelihood: {log_likelihood:.4f}")
        #print(f"Predicted states: {predicted_states}")
        # print(f"True states: {true_states}")
        # print(f"Accuracy: {accuracy:.4f}")

    # return log_likelihoods, accuracies,y_test.tolist()
def main():
    # Generate realistic market data
    print("Generating market simulation...")
    log_likelihoods, predicted_states, true_states, accuracies, results_df = generate_data_and_train_hmm()

    # Evaluate HMM on the data
    metrics = evaluate_hmm(results_df)
    print('\nPrediction Metrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.2%}')

    print('\nPlotting results...')
    plot_market_simulation(results_df)


if __name__ == "__main__":
    main()