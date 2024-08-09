import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SnakeAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeAI, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def evaluate_fitness(snake):
    # Run the snake in the environment and return its fitness score
    # Fitness can be based on score, survival time, etc.
    pass

def crossover(parent1, parent2):
    child = SnakeAI(input_size, hidden_size, output_size)
    for param1, param2, param_child in zip(parent1.parameters(), parent2.parameters(), child.parameters()):
        param_child.data.copy_(0.5 * param1.data + 0.5 * param2.data)
    return child

def mutate(snake, mutation_rate=0.01):
    for param in snake.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param) * 0.1

def genetic_algorithm(population_size, generations, mutation_rate):
    population = [SnakeAI(input_size, hidden_size, output_size) for _ in range(population_size)]
    
    for generation in range(generations):
        fitness_scores = [evaluate_fitness(snake) for snake in population]
        sorted_population = [snake for _, snake in sorted(zip(fitness_scores, population), reverse=True)]
        
        next_generation = sorted_population[:population_size // 2]
        
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(next_generation, 2)
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            next_generation.append(child)
        
        population = next_generation
        print(f"Generation {generation} - Best Fitness: {max(fitness_scores)}")

# Parameters
input_size = 11  # Example input size
hidden_size = 256
output_size = 3
population_size = 50
generations = 100
mutation_rate = 0.01

genetic_algorithm(population_size, generations, mutation_rate)