import numpy as np


class TraidingAiStrategy(object):
    """
    Training Strategy for AI
    """
    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.pop_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.debug = False

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            weights_population.append(weights[index] + (self.sigma * i))
        return weights_population

    def train(self, epoch=100):
        for i in range(epoch):
            population = []
            generation_rewards = np.zeros(self.pop_size)
            for k in range(self.pop_size):
                x = []
                for w in self.weights:
                    # Add random shape to x list
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.pop_size):
                weights_per_population = self._get_weight_from_population(self.weights, population[k])
                generation_rewards[k] = self.reward_function(weights_per_population)
            generation_rewards = (generation_rewards - np.mean(generation_rewards)) / (np.std(generation_rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                array = np.array([p[index] for p in population])
                # .T to take transposed array instead of regular array
                self.weights[index] = (w + self.learning_rate / (self.pop_size * self.sigma) *
                                       np.dot(array.T, generation_rewards).T)
            if self.debug and i % 10 == 0:
                print(f'Epooch {i} with reward: {self.reward_function(self.weights)}')
