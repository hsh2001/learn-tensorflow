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
        hidden_size_list: list[int],
        learning_late=0.01,
        weight_init_std=0.01,
    ) -> None:
        self.learning_late = learning_late
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        hidden_layer_count = len(hidden_size_list)
        self.hidden_layer_count = hidden_layer_count
        self.params = {}
        prev_nodes = input_size

        for i in range(hidden_layer_count):
            nodes = hidden_size_list[i]
            self.params['W' + str(i)] = weight_init_std * \
                np.random.rand(prev_nodes, nodes)
            self.params['b' + str(i)] = np.zeros(nodes)
            prev_nodes = hidden_size_list[i]

        self.params['W' + str(hidden_layer_count)] = weight_init_std * \
            np.random.rand(prev_nodes, output_size)
        self.params['b' + str(hidden_layer_count)] = np.zeros(output_size)

    def predict(self, x):
        y = x
        for i in range(self.hidden_layer_count):
            w = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            y = sigmoid(np.dot(y, w) + b)

        w = self.params['W' + str(self.hidden_layer_count)]
        b = self.params['b' + str(self.hidden_layer_count)]
        y = softmax(np.dot(y, w) + b)

        return y

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

        gradient = {}

        def set_gradient(key: str):
            gradient[key] = numerical_gradient(
                loss_w,
                self.params[key]
            )

        for i in range(self.hidden_layer_count + 1):
            set_gradient('W' + str(i))
            set_gradient('b' + str(i))

        return gradient

    def fit(self, x_train, t_train):
        gradient = self.get_gradient(x_train, t_train)

        for key in gradient.keys():
            self.params[key] -= self.learning_late * gradient[key]

    def mini_batch_fit(self, x_train, t_train, batch_size):
        batch_mask = np.random.choice(self.input_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        prev_loss = self.get_loss(x_batch, t_batch)
        self.fit(x_batch, t_batch)
        print('loss diff :', self.get_loss(x_batch, t_batch) - prev_loss)


(x_train, t_train), (x_test, t_test) = load_mnist(
    normalize=True,
    one_hot_label=True,
)

network = LayerNetwork(
    input_size=x_train.shape[1],
    output_size=t_train.shape[1],
    hidden_size_list=[32, 16],
    learning_late=0.001
)

for i in range(1000):
    network.mini_batch_fit(x_train, t_train, 100)
    if i != 0 and 0 == (i % 20):
        accuracy = 100 * network.get_accuracy(x_test, t_test)
        print('accuracy', accuracy, '%')
