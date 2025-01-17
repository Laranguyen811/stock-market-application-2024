import random

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


