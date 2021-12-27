import numpy as np
import matplotlib.pyplot as plt
import os

def acc(Y, Y_hat):
    pred = np.copy(Y_hat)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    return np.sum(Y == pred) / Y.shape[0]

def plot_loss(losses):
    plt.plot(losses)
    plt.title('loss after each epoch')
    
    if not os.path.isdir('./img'):
        os.mkdir('./img')
    if not os.path.isdir('./img/full'):
        os.mkdir('./img/full')

    plt.savefig(f'./img/full/loss_curve.png')
    plt.close()

def plot_boundary(X, Y, forward, epoch):
    res = 50
    x1, x2 = -1.5, 2.5
    y1, y2 = -1, 1.5
    grid_x, grid_y = np.meshgrid(np.linspace(x1, x2, num=res), np.linspace(y1, y2, num=res))

    grid = np.concatenate((grid_x.reshape((res ** 2, 1)), grid_y.reshape((res ** 2, 1))), axis=1)
    Y_hat = forward(grid)

    num_contours = 2 * 5 + 1
    plt.contourf(grid_x, grid_y, Y_hat.reshape(res, res), np.linspace(0, 1, num=num_contours), alpha=0.7)
    plt.contour(grid_x, grid_y, Y_hat.reshape(res, res), [0.5])
    plt.scatter(X.T[0], X.T[1], c=Y, edgecolors='black')
    plt.title(f'decision boundary after epoch {epoch + 1:5d}')

    if not os.path.isdir('./boundaries'):
        os.mkdir('./boundaries')
    if not os.path.isdir('./boundaries/full'):
        os.mkdir('./boundaries/full')
    
    plt.savefig(f'./boundaries/full/boundary_epoch_{epoch + 1:05d}.png')
    plt.close()
