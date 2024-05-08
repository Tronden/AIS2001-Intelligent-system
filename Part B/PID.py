import random
import numpy as np
import matplotlib.pyplot as plt

class PIDGAConfig:
    def __init__(self, Kp_range=(0, 10), Ki_range=(0, 10), Kd_range=(0, 10), 
                 setpoint=5, generations=50, population_size=20, 
                 crossover_probability=0.7, mutation_rate=0.03):
        self.Kp_range = Kp_range
        self.Ki_range = Ki_range
        self.Kd_range = Kd_range
        self.setpoint = setpoint
        self.generations = generations
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_rate = mutation_rate

    def simulate_pid_response(self, Kp, Ki, Kd, n=100, tau=10):
        """ Simulates a step response for a PID controlled system. """
        dt = 1.0
        y = 0
        integral = 0
        prev_error = 0
        ys = []

        for _ in range(n):
            error = self.setpoint - y
            integral += error * dt
            derivative = (error - prev_error) / dt
            y += (Kp * error + Ki * integral + Kd * derivative - y) / tau * dt
            ys.append(y)
            prev_error = error

        return ys

    def evaluate_fitness(self, response):
        """ Calculate fitness based on steady state error and overshoot. """
        steady_state_error = abs(response[-1] - self.setpoint)
        overshoot = max(0, max(response) - self.setpoint)
        return 1 / (1 + steady_state_error + overshoot)

    def initialize_population(self):
        """ Initializes population with random PID parameters. """
        return [[random.uniform(*self.Kp_range), random.uniform(*self.Ki_range), random.uniform(*self.Kd_range)]
                for _ in range(self.population_size)]

    def tournament_selection(self, population, fitnesses):
        """ Selects the best individual from a random sample. """
        tournament_size = max(2, int(len(population) * 0.1))
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        """ Performs a single point crossover. """
        if random.random() < self.crossover_probability:
            point = random.randint(1, len(parent1) - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        return parent1.copy(), parent2.copy()

    def mutate(self, ind):
        """ Applies mutation to the individual. """
        return [gene + random.uniform(-0.1, 0.1) if random.random() < self.mutation_rate else gene for gene in ind]

    def run_genetic_algorithm(self):
        """ Runs the genetic algorithm process. """
        population = self.initialize_population()
        fitness_history = []

        for generation in range(self.generations):
            responses = [self.simulate_pid_response(*ind) for ind in population]
            fitnesses = [self.evaluate_fitness(response) for response in responses]
            best_index = np.argmax(fitnesses)
            fitness_history.append(fitnesses[best_index])

            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            population = new_population[:self.population_size]

            print(f"Generation {generation}: Best Fitness = {fitnesses[best_index]} Params = {population[best_index]}")

        # Plot the evolution of the best fitness
        plt.plot(fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title('Evolution of Fitness over Generations')
        plt.grid(True)
        plt.show()

# Example usage
config = PIDGAConfig(generations=250, population_size=10)
config.run_genetic_algorithm()