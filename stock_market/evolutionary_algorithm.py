import random
import os
import sys
import importlib.util
from typing import List, Union, Dict
import numpy as np
#src_path = os.path.abspath("src")
#if src_path not in sys.path:
 #   sys.path.insert(0, src_path)

# Add external diagnostic_tool path


spec = importlib.util.spec_from_file_location("strategy_metrics", r"C:\Users\User\diagnostic_tool_workspace\diagnostic_tool\src\diagnostic_tool\strategy_metrics.py")
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)
# Optional: change working directory if needed for file I/O
# os.chdir(module_path)

def normalise(value, min_value, max_value):
    '''
    Takes a value and normalises it between a minimum and maximum value.
    Args:
        value (float): A float number representing the value to be normalised.
        min_value (float): A float number representing the minimum value.
        max_value (float): A float number representing the maximum value.
    Returns:
        float: A float number representing the normalised value.
    '''
    return (value - min_value) / (max_value - min_value) if max_value != min_value else 0.0 # min-max normalisation, ensuring values range from 0 to 1, preserving relative distances and distribution shapes
def evaluate_strategies(
        returns: Union[List[float], np.ndarray],
        risk_free_rate: Union[float, np.ndarray],threshold: float, equity_curve:Union[List[float], np.ndarray],
        verbose: bool = True
) -> Dict[str, object]:
    '''
    Takes a list of returns and evaluates the strategies based on various financial metrics.
    Args:
        returns (list): A list of returns.
        risk_free_rate (float): A float number representing the risk-free rate. Defaults to 0.0.
        threshold (float): A float number representing the threshold for evaluation. Defaults to 0.5.
        equity_curve (Union[List[float], np.ndarray]): The equity curve of the strategy (list or np.ndarray).
    Returns:
        dict: A dictionary containing the evaluation results.
    '''
    return {
        'Sharpe Ratio': strategy.check_sharpe_ratio(returns,risk_free_rate,threshold),
        'Sortino Ratio': strategy.check_sortino_ratio(returns, risk_free_rate, threshold),
        'Calmar Ratio': strategy.check_calmar_ratio(returns, risk_free_rate, threshold),
        'Maximum Drawdown': strategy.check_max_drawdown(equity_curve),
        'Omega Ratio': strategy.check_omega_ratio(returns, risk_free_rate, threshold),

    }
# Example usage:
evaluate_strategies([0.1,0.2,0.3], 0.2,0.3,[1000, 1100, 1200, 1050, 1150, 1250])

def weighted_strategy_scores(scores,weights):
    '''
    Takes a list of scores and weights and returns a weighted score.
    Args:
        scores (list): A list of scores.
        weights (list): A list of weights.
    Returns:
        float: A float number representing the weighted score.
    '''
    score = 0.0
    for name,result in scores.items():
        value = score['name']
        weight = weights.get(name,0)
        normalised_value = normalise(value,0,1)  # Normalising the value between 0 and 1
        weighted_value = normalised_value * weight
        score += weighted_value  # Summing up the weighted values
    return score
def fitness(individual,data:dict):
    '''
    Takes a dictionary historical data and the individual's trading strategy and returns a fitness score.
    Inputs:
        data(dict): A dictionary with historical data
        individual(int): An integer representing individual's trading strategy
    Returns:
        float: A float number representing fitness score
    '''
    return score
def initialise_population(population_size, param_ranges):
    ''' Takes the population and creates a population.
    Args:
        population_size (int): An integer representing the population size.
        param_ranges (int): An integer representing the parameter ranges.
    Returns:
         list: A list representing population

    '''
    population = []
    for _ in range(population_size):  # The underscore _ conveys that the variable is not needed in the loop body
        individual = {param: random.uniform(*range_) for param,range_ in param_ranges.items()}  # random.uniform generates random floating points
        population.append(individual)
    return population

def select(population, fitness_scores):
    '''
    Takes a string of population and a dictionary of fitness scores and returns a selection.
    Inputs:
        population (list): A list representing population
        fitness_scores (dict): A dictionary containing fitness scores for each individual
    Returns:
        list: A list representing selection
    '''
    population = []
    selection = random.choices(population, weights=fitness_scores,k=len(population))  # Select randomly from the population list
    return selection

def crossover(parent1, parent2):
    '''
    Takes a string of population and a dictionary of fitness scores and returns a crossover (combining pairs of individuals to create offsprings).
    Inputs:
        parent1 (list): A list representing population of the first parent
        parent2 (list): A list representing population of the second parent
    Returns:
        float: A float number representing crossover
        float: A float number representing crossover
    '''
    crossover = []
    crossover_point = random.randint(0,len(parent1) - 1)  # Select a random integer
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1,child2

def mutate(individual, mutation_rate):
    '''
    Takes a string of population and a dictionary of mutation rates and returns a mutation (maintaining genetic diversity and helping to overcome suboptimal solutions.
    Inputs:
        individual (list): A list representing population
        mutation_rate (float): A float representing the mutation rate
    Returns:
        str: A string representing mutation rate
    '''
    for i in range(0,len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(-1,1)  #random.uniform(a,b) returns a value x so that a<= x <= b
        return individual

def evolution_algorithm(data,pop_size,num_generations,num_genes,mutation_rate):
    '''
    Takes a dictionary historical data, a population size, a number of generations, a number of genes and a mutation rate and returns the best individual.
    Inputs:
        data(dict): A dictionary of historical data
        pop_size(int): An integer representing the population size
        num_generations(int): An integer representing the number of generations
        num_genes(int): An integer representing the number of genes
        mutation_rate(float): A float representing the mutation rate
    Returns:
        float: A float number representing the best individual
    '''
    population = initialise_population(pop_size,num_genes)
    for generation in range(num_generations):
        fitness_scores = [fitness(i,data) for i in population]
        selection = select(population, fitness_scores)
        next_generation = []
        for i in range(0,len(selection),2):
            parent1,parent2 = crossover(selection[i],selection[i+1])
            child1,child2 = mutate(parent1,parent2)
            next_generation.append(mutate(child1,mutation_rate))
            next_generation.append(mutate(child2,mutation_rate))
        population = next_generation
    best_individual = max(population,key=lambda i:fitness(i,data))
    return best_individual


