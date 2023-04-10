from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class AdalineStochastic:
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        if y.ravel().shape[0] <= 1:
            self._update_weights(X, y)
            return self

        for xi, target in zip(X, y):
            self._update_weights(xi, target)
        return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss

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

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

    ada1 = AdalineStochastic(n_iter=15, eta=0.01, random_state=1)
    ada1.fit(X_std, y)

    ada2 = AdalineStochastic(n_iter=15, eta=0.01, random_state=1)
    ada2.fit(X_std, y)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    plt.tight_layout()
    ax[0].set_title('Adaline - Stochastic gradient descent')
    ax[0].legend(loc='upper left')
    ax[0].plot(range(1, len(ada1.losses_) + 1), ada1.losses_, marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Average loss')

    ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Average loss')
    plt.show()
