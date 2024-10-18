import numpy as np
from standard_nn import create_standard_nn
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        neurons_layer1 = np.random.randint(32, 128)
        neurons_layer2 = np.random.randint(32, 128)
        learning_rate = 10 ** np.random.uniform(-4, -1)


        optimizers = [Adam, SGD, RMSprop]
        selected_optimizer = np.random.choice(optimizers)
        optimizer = selected_optimizer(learning_rate=learning_rate)

        epochs = np.random.randint(10, 500)

        individual = {'neurons_layer1': neurons_layer1,
                      'neurons_layer2': neurons_layer2,
                      'learning_rate': learning_rate,
                      'optimizer': optimizer,
                      'epochs': epochs}
        population.append(individual)

    return population


def evaluate_fitness(individual, X_train, y_train, X_val, y_val):
    model = create_standard_nn(num_features=X_train.shape[1], num_classes=y_train.shape[1])
    model.layers[0].units = individual['neurons_layer1']
    model.layers[1].units = individual['neurons_layer2']
    model.compile(optimizer=individual['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=individual['epochs'], verbose=0, validation_data=(X_val, y_val))
    val_accuracy = history.history['val_accuracy'][-1]

    return val_accuracy


def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        if np.random.rand() < 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child


def mutate(individual):
    mutation_rate = 0.1
    for key in individual:
        if np.random.rand() < mutation_rate:
            if key.startswith('neurons'):
                individual[key] = np.random.randint(32, 128)
            elif key == 'learning_rate':
                individual[key] = 10 ** np.random.uniform(-4, -1)
            elif key == 'optimizer':
                # Mutate optimizer to a randomly selected one
                optimizers = [Adam, SGD, RMSprop]
                selected_optimizer = np.random.choice(optimizers)
                individual[key] = selected_optimizer(learning_rate=individual['learning_rate'])
            elif key == 'epochs':
                individual[key] = np.random.randint(10, 500)
    return individual


def train_and_optimize_with_genetic_algorithm(X_train, y_train, X_val, y_val, population_size, generations):
    population = initialize_population(population_size)

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        fitness_scores = [evaluate_fitness(ind, X_train, y_train, X_val, y_val) for ind in population]

        best_index = np.argmax(fitness_scores)
        best_individual = population[best_index]


        new_population = [best_individual]  # Elitism, keeping the best individual
        for _ in range(population_size - 1):
            parent1 = population[np.random.choice(population_size)]
            parent2 = population[np.random.choice(population_size)]
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    best_params = population[best_index]
    return best_params
