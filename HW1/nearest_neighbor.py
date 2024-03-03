import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

def get_data(train_size=10000, test_size=1000):
    # Load dataset
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    # Generate the indices to sample from
    idx = np.random.RandomState(0).choice(70000, train_size + test_size)

    # Split into train and test
    train = data[idx[:train_size], :].astype(int)
    train_labels = labels[idx[:train_size]].astype(int)
    test = data[idx[train_size:], :].astype(int)
    test_labels = labels[idx[train_size:]].astype(int)

    return train, train_labels, test, test_labels

# Question a
def knn(train, train_labels, query, k):
    distances = np.linalg.norm(train - query, axis=1)
    sorted_indices = np.argsort(distances)
    top_k_indices = sorted_indices[:k]
    top_k_labels = train_labels[top_k_indices]
    return np.argmax(np.bincount(top_k_labels))

# Question b
def test_knn(train, train_labels, test, test_labels, n=1000, k=10):
    y_hats = [knn(train[:n], train_labels[:n], query, k) for query in test]
    mean = (y_hats == test_labels).mean()
    print(f"Accuracy: {mean}")
    return mean

# Question c
def test_and_plot_knn_for_ks(train, train_labels, test, test_labels):
    ys = []
    ks = range(1, 101)
    for k in ks:
        ys.append(test_knn(train, train_labels, test, test_labels, 1000, k))

    plt.plot(list(ks), ys) 
    plt.title('Accuracy vs K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()

# Question d
def test_and_plot_knn_for_ns(train, train_labels, test, test_labels):
    ys = []
    ns = range(100, 5001, 100)
    for n in ns:
        ys.append(test_knn(train, train_labels, test, test_labels, n, 1))

    plt.plot(list(ns), ys) 
    plt.title('Accuracy vs Number of Training Samples')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    train, train_labels, test, test_labels = get_data()
    # test_knn(train, train_labels, test, test_labels, 1000, 10)
    # test_and_plot_knn_for_ks(train, train_labels, test, test_labels)
    # test_and_plot_knn_for_ns(train, train_labels, test, test_labels)