import random

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

#def evaluate_fitness(individual,)
