import random
import string
import matplotlib.pyplot as plt

# Define target string
target_string = "Trond_Zachariassen*566107"

# Lists to store fitness scores and strings from each generation
Scores = []
Strings = []

# Genetic algorithm parameters
population_size = 2000
mutation_rate = 0.01
max_generations = 1000
tournament_size = 5

# Allowed characters
allowed_characters = string.ascii_letters + string.digits + "_*"

# Generate random strings
def generate_individual():
    return ''.join(random.choice(allowed_characters) for _ in range(len(target_string)))

# Calculating fitness
def fitness(individual):
    score = sum(1 for i, j in zip(individual, target_string) if i == j)
    return score / len(target_string)

# Tournament selection
def tournament_selection(population, tournament_size):
    tournament = random.choices(population, k=tournament_size)
    return max(tournament, key=fitness)

# Single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# Random character mutation
def mutate(individual, mutation_rate):
    mutated_individual = ''
    for char in individual:
        if random.random() < mutation_rate:
            mutated_individual += random.choice(allowed_characters)
        else:
            mutated_individual += char
    return mutated_individual

# Initial population
population = [generate_individual() for _ in range(population_size)]

# Evolution loop
for generation in range(max_generations):
    # Evaluate fitness of each individual
    fitness_scores = [fitness(individual) for individual in population]
    
    # Find best fit individual
    best_fit_index = fitness_scores.index(max(fitness_scores))
    best_fit_individual = population[best_fit_index]
    best_fitness = fitness(best_fit_individual)
    print("Best Fitness in generation:", generation + 1, "String:", best_fit_individual, "Score:", best_fitness)
    Scores.append(best_fitness)
    Strings.append(best_fit_individual)

    # Terminate if solution is found
    if best_fit_individual == target_string:
        break
    
    # Selection
    new_population = [tournament_selection(population, tournament_size) for _ in range(population_size)]
    
    # Crossover
    for i in range(0, population_size, 2):
        parent1 = new_population[i]
        parent2 = new_population[i + 1]
        child1, child2 = crossover(parent1, parent2)
        new_population[i] = child1
        new_population[i + 1] = child2
    
    # Mutation
    population = [mutate(individual, mutation_rate) for individual in new_population]

# Printing the best score and string.
best_generation = Scores.index(max(Scores))
print("Best generation:", best_generation + 1, "String:", Strings[best_generation], "Score:", Scores[best_generation])

# Plotting the fitness scores from each generation
plt.plot(Scores)
plt.title('Fitness Scores Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness Score')
plt.grid(True)
plt.show()