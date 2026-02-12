"""
Coded By Munaf Studios on 27 January 2026
"""
import numpy as np
import matplotlib.pyplot as plt


def create_dataset(N, K=2):
    N=100
    D=2
    X = np.zeros((N* K, D))
    y = np.zeros((N* K))

    for j in range(K):
        ix = range(N * j , N * (j + 1 ))
        r = np.linspace(0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4 , N) + np.random.rand(N) * 0.02
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    plt.scatter(X[:, 0] , X[:,1], c=y, s = 40, cmap = plt.cm.Spectral)
    plt.show()

    return X, y

def plot_counter(X, y, model, parameters):
    h = 0.02
    x_min, x_max = X[:, 0].min() -1 , X[:,0].max() +1
    y_min, y_max = X[:, 1].min() -1 , X[:,1].max() +1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    _, Z = model.forward_prop(points, parameters)


    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)


    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    plt.savefig('spiral_net.png')