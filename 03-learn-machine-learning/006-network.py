from data.mnist import load_mnist
import numpy as np


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    max_item = np.max(a)
    new_a = max_item - a
    exp_a = np.exp(new_a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))


class LayerNetwork:
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        learning_late=0.01,
        weight_init_std=0.01,
    ) -> None:
        self.learning_late = learning_late
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.params = {
            'W1': weight_init_std * np.random.rand(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.rand(hidden_size, output_size),
            'b2': np.zeros(output_size),
        }

    def predict(self, x):
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        return z2

    def get_loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def get_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def get_gradient(self, x, t):
        def loss_w(w): return self.get_loss(x, t)
        return {
            'W1': numerical_gradient(loss_w, self.params['W1']),
            'b1': numerical_gradient(loss_w, self.params['b1']),
            'W2': numerical_gradient(loss_w, self.params['W2']),
            'b2': numerical_gradient(loss_w, self.params['b2']),
        }

    def fit(self, x_train, t_train):
        gradient = self.get_gradient(x_train, t_train)

        for key in ('W1', 'b1', 'W2', 'b2'):
            self.params[key] -= self.learning_late * gradient[key]

    def mini_batch_fit(self, x_train, t_train, batch_size):
        batch_mask = np.random.choice(self.input_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        print('loss', network.get_loss(x_batch, t_batch))
        self.fit(x_batch, t_batch)


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,
    one_hot_label=True,
)

network = LayerNetwork(
    input_size=x_train.shape[1],
    output_size=t_train.shape[1],
    hidden_size=10,
)

for i in range(1000):
    print(i, end=' ')
    network.mini_batch_fit(x_train, t_train, 100)
    if 0 == (i % 10):
        print('accuracy', 100 * network.get_accuracy(x_test, t_test), '%')
