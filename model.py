import numpy as np
from utils.debug import acc, plot_boundary, plot_loss

class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

        self.init_parameters()
        
    def init_parameters(self):
        self.parameters = {}

        for i, layer in enumerate(self.layers):
            self.parameters[f'W{i + 1}'] = np.random.randn(layer['in_channels'], layer['out_channels'])
            self.parameters[f'b{i + 1}'] = np.random.randn(layer['out_channels'])

    def forward(self, X, training=False):
        cache = {}

        for i, layer in enumerate(self.layers):
            W, b = self.parameters[f'W{i + 1}'], self.parameters[f'b{i + 1}']
            A_prev = X if i == 0 else cache[f'A{i}']

            Z = A_prev @ W + b
            A = layer['activation'].evaluate(Z)

            cache[f'Z{i + 1}'], cache[f'A{i + 1}'] = Z, A

        if training:
            self.cache = cache

        return A

    def backward(self, X, Y):
        self.gradients = {}

        for i, layer in reversed(list(enumerate(self.layers))):
            if i == len(self.layers) - 1:
                A = self.cache[f'A{i + 1}']
                dA = self.loss.derivative(Y, A)
            else:
                dZ_next = self.gradients[f'dZ{i + 2}']
                W_next = self.parameters[f'W{i + 2}']
                dA = dZ_next @ W_next.T

            Z = self.cache[f'Z{i + 1}']
            dZ = dA * layer['activation'].derivative(Z)

            self.gradients[f'dA{i + 1}'], self.gradients[f'dZ{i + 1}'] = dA, dZ

            A_prev = X if i == 0 else self.cache[f'A{i}']
            N = Y.shape[0]

            dW = (A_prev.T @ dZ) / N
            db = np.sum(dZ, axis=0) / N

            self.gradients[f'dW{i + 1}'], self.gradients[f'db{i + 1}'] = dW, db

    def train(self, X, Y, n_iterations, alpha):
        self.forward(X, training=True)
        losses = []

        for epoch in range(n_iterations):
            self.backward(X, Y)
            
            for i in range(len(self.layers)):
                self.parameters[f'W{i + 1}'] -= alpha * self.gradients[f'dW{i + 1}']
                self.parameters[f'b{i + 1}'] -= alpha * self.gradients[f'db{i + 1}']

            Y_hat = self.forward(X, training=True)

            epoch_loss = np.sum(self.loss.evaluate(Y, Y_hat))
            accuracy = acc(Y, Y_hat)
            print(f'epoch {epoch + 1:5d}: loss = {epoch_loss:6f}, accuracy = {accuracy:6f}')

            if (epoch + 1) % 100 == 0:
                plot_boundary(X, Y, self.forward, epoch)

            losses.append(epoch_loss)

        plot_loss(losses)
        


