"""
Optimizers return updated weight receiving original weights and gradients
"""
import numpy as np


class SGD:
    def __init__(self):
        pass

    def update(self, weight, gradient, learning_rate):
        updated_weight = w - learning_rate * gradient

        return updated_weight


class Momentum:
    def __init__(self, gamma):
        self.gamma = gamma
        self.velocity = 0

    def update(self, weight, gradient, learning_rate):
        self.velocity = self.gamma * self.velocity + (1 - self.gamma) * gradient
        updated_weight = w - self.velocity * learning_rate

        return updated_weight


class RMSProp:
    def __init__(self, gamma, epsilon):
        self.epsilon = epsilon
        self.gamma = gamma
        self.s = 0

    def update(self, weight, gradient, learning_rate):
        self.s = self.gamma * self.s + (1 - self.gamma) * gradient ** 2
        updated_weight = w - learning_rate * (gradient / ((self.s) ** 0.5 + self.epsilon))

        return updated_weight
