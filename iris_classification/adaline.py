
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Adaline:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


if __name__ == '__main__':
    try:
        df = pd.read_csv('./iris.data', header=None, encoding='utf-8')
    except:
        print('Download iris.data from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
        exit(1)

    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

    ada1 = Adaline(n_iter=15, eta=0.1).fit(X, y)
    ada2 = Adaline(n_iter=15, eta=0.0001).fit(X, y)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].plot(range(1, len(ada1.losses_) + 1),
               np.log10(ada1.losses_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Mean squared error)')
    ax[0].set_title('Adaline - Learning rate 0.1')
    ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Mean squared error')
    ax[1].set_title('Adaline - Learning rate 0.0001')
    plt.show()
