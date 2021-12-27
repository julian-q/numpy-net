import numpy as np

class Activation:
    def evaluate(self, Z):
        raise NotImplementedError

    def derivative(self, Z):
        raise NotImplementedError

class ReLU(Activation):
    def evaluate(self, Z):
        return np.maximum(0, Z)

    def derivative(self, Z):
        D = np.copy(Z)
        D[D < 0] = 0
        D[D > 0] = 1

        return D

class Sigmoid(Activation):
    def evaluate(self, Z):
        return 1 / (1 + np.exp(-Z))

    def derivative(self, Z):
        return self.evaluate(Z) * (1 - self.evaluate(Z))
        