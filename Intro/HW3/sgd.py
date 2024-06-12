import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """

    w = np.array([0] * len(data[0]))

    for t in range(1, T+1):
        i = np.random.randint(0, len(data))
        yi = labels[i]
        xi = data[i]
        eta = eta_0 / t

        if np.dot(w, xi) * yi < 1:
            w = (1-eta)*w + eta*C*yi*xi
        else:
            w = (1-eta)*w

    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w = np.array([0] * len(data[0]))
    
    for t in range(1, T+1):
        i = np.random.randint(0, len(data))
        yi = labels[i]
        xi = data[i]
        eta = eta_0 / t

        w = w - eta * ((-xi * yi) / (1 + np.exp(yi * np.dot(w, xi))))

    return w

def assess_accuracy(data, labels, w):
    return np.mean(np.sign(np.dot(data, w)) == labels)


def get_best_eta0_hinge():
    T = 1000
    C = 1
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    rangers = np.linspace(-5, 6, 100)

    etas = [10**i for i in rangers]
    accuracies = []

    for eta in tqdm(etas):
        accuracy = 0
        for _ in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta, T)
            accuracy += assess_accuracy(validation_data, validation_labels, w)

        accuracies.append(accuracy / 10)

    plt.plot(etas, accuracies)    
    plt.xscale('log')
    plt.xlabel('η0')
    plt.ylabel('Average accuracy on validation set')
    plt.title('Average accuracy on validation set as a function of η0')
    plt.show()

    return etas[np.argmax(accuracies)]



def get_best_C_hinge(eta0):
    T = 1000
    train_data, train_labels, validation_data, validation_labels, _, _ = helper()

    rangers = np.linspace(-5, 6, 100)

    Cs = [10**i for i in rangers]
    accuracies = []

    for C in tqdm(Cs):
        accuracy = 0
        for _ in range(10):
            w = SGD_hinge(train_data, train_labels, C, eta0, T)
            accuracy += assess_accuracy(validation_data, validation_labels, w)

        accuracies.append(accuracy / 10)

    plt.plot(Cs, accuracies)    
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Average accuracy on validation set')
    plt.title(f'Average accuracy on validation set as a function of C (η0 = {round(eta0, 4)})')
    plt.show()

    return Cs[np.argmax(accuracies)]

def get_best_classifier_hinge(eta0, C):
    T = 20000
    train_data, train_labels, _, _, _, _ = helper()

    w = SGD_hinge(train_data, train_labels, C, eta0, T)

    plt.imshow(w.reshape((28, 28)), interpolation="nearest")
    plt.title(f'w as an image (η0 = {round(eta0, 4)}, C = {round(C, 4)}, T = {T})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

    return w

def get_accuracy_of_classifier(w):
    _, _, _, _, test_data, test_labels = helper()
    return assess_accuracy(test_data, test_labels, w)

def get_best_eta0_log():
    T = 1000
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    rangers = np.linspace(-5, 6, 100)

    etas = [10**i for i in rangers]
    accuracies = []

    for eta in tqdm(etas):
        accuracy = 0
        for _ in range(10):
            w = SGD_log(train_data, train_labels, eta, T)
            accuracy += assess_accuracy(validation_data, validation_labels, w)

        accuracies.append(accuracy / 10)

    plt.plot(etas, accuracies)    
    plt.xscale('log')
    plt.xlabel('η0')
    plt.ylabel('Average accuracy on validation set')
    plt.title('Average accuracy on validation set as a function of η0')
    plt.show()

    return etas[np.argmax(accuracies)]


def get_best_classifier_log(eta0):
    T = 20000
    train_data, train_labels, _, _, _, _ = helper()

    w = SGD_log(train_data, train_labels, eta0, T)

    plt.imshow(w.reshape((28, 28)), interpolation="nearest")
    plt.title(f'w as an image (η0 = {round(eta0, 6)}, T = {T})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()

    return w

def SGD_log_iterations(eta_0):
    T = 20000
    data, labels, _, _, _, _ = helper()
    w = np.array([0] * len(data[0]))
    
    norms = []
    for t in range(1, T+1):
        i = np.random.randint(0, len(data))
        yi = labels[i]
        xi = data[i]
        eta = eta_0 / t

        w = w - eta * ((-xi * yi) / (1 + np.exp(yi * np.dot(w, xi))))

        norms.append(np.linalg.norm(w))

    plt.plot(norms)
    plt.xlabel('Iterations')
    plt.ylabel('||w||')
    plt.title(f'Norm of w as a function of iterations (η0 = {round(eta_0, 6)})')
    plt.show()


eta0 = get_best_eta0_log()
w = get_best_classifier_log(eta0)
print(get_accuracy_of_classifier(w))
SGD_log_iterations(eta0)


eta0 = get_best_eta0_hinge()
C = get_best_C_hinge(eta0)
w = get_best_classifier_hinge(eta0, C)
print(get_accuracy_of_classifier(w))
