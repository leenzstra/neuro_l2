import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


act_dict = {
    "sigmoid": {
        "func": sigmoid, "derivative": sigmoid_derivative
    },
    "relu": {
        "func": relu, "derivative": relu_derivative
    }
}


class NN():
    def __init__(self, layers, activations=[]):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activations = activations

        assert len(layers) == len(
            activations), "Number of layers and activations should be equal"

        self._setup()

    def _setup(self):
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.randn(
                self.layers[i+1], self.layers[i]))
            self.biases.append(np.zeros((1, self.layers[i+1])))

    def forward_propagation(self, x):
        a = np.copy(x)
        Zs = []
        As = [a]

        for i in range(len(self.weights)):
            z = np.dot(As[-1], self.weights[i].T) + self.biases[i]
            a = act_dict[self.activations[i]]["func"](z)

            Zs.append(z)
            As.append(a)

        return (Zs, As)

    def backward_propagation(self, y, z_s, a_s):
        dw = []  # d/dW
        db = []  # d/dB
        deltas = [None] * len(self.weights)

        # последний слой
        err = y - a_s[-1]
        deltas[-1] = err

        # остальные слои
        for i in reversed(range(len(deltas)-1)):
            deltas[i] = np.dot(deltas[i+1], self.weights[i+1]
                               ) * act_dict[self.activations[i]]["derivative"](z_s[i])
            
        # print(deltas)

        db = [d for d in deltas]
        dw = [np.dot(d.T, a_s[i]) for i, d in enumerate(deltas)]

        return dw, db

    def train(self, Xs, Ys, epochs=100, lr=0.01, accuracy=0.0001):
        history = {'loss':[0] * epochs}

        for e in range(epochs):
            epoch_loss = 0
            for x, y in zip(Xs, Ys):
                x = np.array(x).reshape((1, len(x)))
                y = np.array(y).reshape((1, len(y)))
                z_s, a_s = self.forward_propagation(x)
                dw, db = self.backward_propagation(y, z_s, a_s)

                self.weights = [w + lr*dweight for w,
                                dweight in zip(self.weights, dw)]
                self.biases = [b + lr*dbias for b,
                               dbias in zip(self.biases, db)]
                
                loss = np.sum((a_s[-1]-y) ** 2)
                epoch_loss += loss

            epoch_loss /= len(Xs)
            print("epoch = {} loss = {}".format(e, epoch_loss))
            history["loss"][e] = epoch_loss

            if loss < accuracy:
                print("accuracy = {}".format(loss))
                break
            
        return history

    def predict(self, x):
        _, a_s = self.forward_propagation(x)
        return a_s[-1][0]
