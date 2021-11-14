import numpy as np


class Model:
    def __init__(self) -> None:
        self.weight = np.random.random()
        self.bias = np.random.random()

    def predict(self, x, weight=0, bias=0):
        if weight == 0:
            weight = self.weight

        if bias == 0:
            bias = self.bias

        return weight * x + bias

    def get_loss(self, x, t, weight=0, bias=0):
        return np.sqrt(np.sum((self.predict(x, weight=weight, bias=bias) - t) ** 2))

    def diff_weight(self, x, t):
        h = 1e-4
        return (
            self.get_loss(x, t, weight=self.weight + h) -
            self.get_loss(x, t, weight=self.weight - h)
        ) / (h * 2)

    def diff_bias(self, x, t):
        h = 1e-4
        return (
            self.get_loss(x, t, bias=self.bias - h) -
            self.get_loss(x, t, bias=self.bias + h)
        ) / (h * 2)

    def fit(self, x, t):
        learning_late = 0.000001
        self.weight = self.weight - learning_late * self.diff_weight(x, t)
        self.bias = self.bias - learning_late * self.diff_bias(x, t)
        loss = self.get_loss(x, t)
        print('loss :', loss)


x = np.array([20, 21, 22, 23, 24])
t = np.array([40, 42, 44, 46, 48])
model = Model()

for i in range(50000):
    model.fit(x, t)

print(model.predict(30))
