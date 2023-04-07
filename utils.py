import numpy as np
import os
import gzip
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt


def test(model, test_data):
    res = []
    for input, label in test_data:
        output = model.forward(input)
        res.append(np.argmax(output) == np.argmax(label))
    accuracy = sum(res) / 100.0
    print(f"****Test accuracy {accuracy} %.")


def fit(model, optimizer, training_data, validation_data, epochs):
    best_accuracy = 0
    train_losses = []
    validate_losses = []
    accuracies = []
    for epoch in range(epochs):
        validate_loss = 0
        res = []
        for input, label in validation_data:
            output = model.forward(input)
            validate_loss += np.where(label == 1, -np.log(output), 0).sum()
            res.append(np.argmax(output) == np.argmax(label))
        validate_loss /= len(validation_data)
        validate_losses.append(validate_loss)
        accuracy = sum(res) / 100.0
        accuracies.append(accuracy)
        random.shuffle(training_data)
        batches = [training_data[k:k + optimizer.batch_size] for k in
                   range(0, len(training_data), optimizer.batch_size)]
        train_loss = 0
        for batch in batches:
            optimizer.zero_grad()
            for input, label in batch:
                output = model.forward(input)
                train_loss += np.where(label == 1, -np.log(output), 0).sum()

                loss_gradient = output - label
                delta_nabla_b, delta_nabla_w = model.backward(loss_gradient)
                optimizer.update(delta_nabla_b, delta_nabla_w)
            optimizer.step()
        train_loss /= len(training_data)
        train_losses.append(train_loss)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save(f"model_{model.sizes[1]}_{optimizer.lr}_{optimizer.weight_decay}.npz")
        print(f"****Epoch {epoch + 1}, accuracy {accuracy} %.")
    data = {
        "train_loss": train_losses,
        "validate_loss": validate_losses,
        "validate_accuracy": accuracies
    }
    pd.DataFrame(data).to_csv(f'logs/log_{model.sizes[1]}_{optimizer.lr}_{optimizer.weight_decay}.csv', )
    return best_accuracy


def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


def load_mnist():
    data_file = gzip.open(os.path.join(os.curdir, "datasets", "mnist.pkl.gz"), "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [vectorized_result(y) for y in train_data[1]]
    train_data = list(zip(train_inputs, train_results))

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [vectorized_result(y) for y in val_data[1]]
    val_data = list(zip(val_inputs, val_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_results = [vectorized_result(y) for y in test_data[1]]
    test_data = list(zip(test_inputs, test_results))

    return train_data, val_data, test_data


def visualize(model, log):
    log[['train_loss', 'validate_loss']].plot(title='train/validate loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("figs/loss.png")
    plt.close()

    log[['validate_accuracy']].plot(title='validate accuracy')
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("figs/accuracy.png")
    plt.close()

    layer1_weights = model.weights[1].flatten().tolist()
    plt.hist(layer1_weights, bins=100)
    plt.title("layer1 weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("figs/layer1_weights.png")
    plt.close()

    layer2_weights = model.weights[2].flatten().tolist()
    plt.hist(layer2_weights, bins=30)
    plt.title("layer2 weights")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("figs/layer2_weights.png")
    plt.close()

    layer1_biases = model.biases[1].flatten().tolist()
    plt.hist(layer1_biases, bins=10)
    plt.title("layer1 biases")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("figs/layer1_biases.png")
    plt.close()

    layer2_biases = model.biases[2].flatten().tolist()
    plt.hist(layer2_biases, bins=10)
    plt.title("layer2 biases")
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig("figs/layer2_biases.png")
    plt.close()
