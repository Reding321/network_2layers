import pandas as pd
import utils
import numpy as np
import os


# ----------------------------------------------
# 定义所需要的函数
def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def sigmoid_gradient(input):
    return sigmoid(input) * (1 - sigmoid(input))


def relu(input):
    return np.maximum(0, input)


def relu_gradient(input):
    return input > 0


def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))


# -----------------------------------
# 网络搭建
class Net:

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [np.array([0])] + [np.random.randn(y, x) / np.sqrt(x) for y, x in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.array([0])] + [np.random.randn(x, 1) for x in sizes[1:]]
        self.linear_transforms = [np.zeros(bias.shape) for bias in self.biases]
        self.activations = [np.zeros(bias.shape) for bias in self.biases]

    def forward(self, input):
        self.activations[0] = input
        for i in range(1, self.num_layers):
            self.linear_transforms[i] = self.weights[i].dot(self.activations[i - 1]) + self.biases[i]
            if i == self.num_layers - 1:
                self.activations[i] = softmax(self.linear_transforms[i])
            else:
                self.activations[i] = relu(self.linear_transforms[i])
        return self.activations[-1]

    def backward(self, loss_gradient):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        nabla_b[-1] = loss_gradient
        nabla_w[-1] = loss_gradient.dot(self.activations[-2].transpose())

        for layer in range(self.num_layers - 2, 0, -1):
            loss_gradient = np.multiply(
                self.weights[layer + 1].transpose().dot(loss_gradient),
                relu_gradient(self.linear_transforms[layer])
            )
            nabla_b[layer] = loss_gradient
            nabla_w[layer] = loss_gradient.dot(self.activations[layer - 1].transpose())

        return nabla_b, nabla_w

    def save(self, filename):
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            linear_transforms=self.linear_transforms,
            activations=self.activations
        )

    def load(self, filename):
        npz_members = np.load(os.path.join(os.curdir, 'models', filename), allow_pickle=True)

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])

        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)

        self.linear_transforms = list(npz_members['linear_transforms'])
        self.activations = list(npz_members['activations'])


# ----------------------------------------
# 定义优化器

class SGD:
    def __init__(self, model, learning_rate, weight_decay, batch_size):
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.nabla_b = [np.zeros(bias.shape) for bias in self.model.biases]
        self.nabla_w = [np.zeros(weight.shape) for weight in self.model.weights]

    def zero_grad(self):
        self.nabla_b = [np.zeros(bias.shape) for bias in self.model.biases]
        self.nabla_w = [np.zeros(weight.shape) for weight in self.model.weights]

    def update(self, delta_nabla_b, delta_nabla_w):
        self.nabla_b = [nb + dnb for nb, dnb in zip(self.nabla_b, delta_nabla_b)]
        self.nabla_w = [nw + dnw for nw, dnw in zip(self.nabla_w, delta_nabla_w)]

    def step(self):
        self.model.weights = [(1 - self.lr * self.weight_decay) * w - (self.lr / self.batch_size) * dw for w, dw in
                              zip(self.model.weights, self.nabla_w)]
        self.model.biases = [(1 - self.lr * self.weight_decay) * b - (self.lr / self.batch_size) * db for b, db in
                             zip(self.model.biases, self.nabla_b)]


# -----------------------------------------------------
# 训练
np.random.seed(42)

batch_size = 16
epochs = 20
layers = [[784, 20, 10], [784, 30, 10], [784, 40, 10]]
learning_rates = [5e-3, 1e-2, 2e-2]
weight_decaies = [0, 1e-2, 2e-2]

print('Loading dataset...')
train_data, val_data, test_data = utils.load_mnist()

print('Training models...')
# 参数查找
best_config = {'accuracy': 0}
for layer in layers:
    for learning_rate in learning_rates:
        for weight_decay in weight_decaies:
            print(
                f"**Current layer: {layer}, Current learning rate: {learning_rate}, Current weight decay: {weight_decay}")
            model = Net(layer)
            optimizer = SGD(model, learning_rate, weight_decay, batch_size)
            accuracy = utils.fit(model, optimizer, train_data, val_data, epochs)
            if accuracy > best_config['accuracy']:
                best_config['accuracy'] = accuracy
                best_config['layer'] = layer
                best_config['learning_rate'] = learning_rate
                best_config['weight_decay'] = weight_decay

# best_config = {'accuracy': 96.77,
# 'layer': [784, 40, 10],
# 'learning_rate': 0.02,
# 'weight_decay': 0}

print("Testing...")
model = Net(best_config['layer'])
model.load(f"model_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.npz")
utils.test(model, test_data)

print("Visualizing...")
log = pd.read_csv(
    f"logs/log_{best_config['layer'][1]}_{best_config['learning_rate']}_{best_config['weight_decay']}.csv")
utils.visualize(model, log)
