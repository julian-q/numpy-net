import numpy as np

class Loss:
    def evaluate(self, Y, Y_hat):
        raise NotImplementedError

    def derivative(self, Y, Y_hat):
        raise NotImplementedError

class LSE(Loss):
    def evaluate(self, Y, Y_hat):
        return ((Y - Y_hat) ** 2) / 2

    def derivative(self, Y, Y_hat):
        return Y_hat - Y

class NLL(Loss):
    def evaluate(self, Y, Y_hat):
        return -(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

    def derivative(self, Y, Y_hat):
        return -((Y / Y_hat) - ((1 - Y) / (1 - Y_hat)))