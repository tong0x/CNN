import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from myLayers import *
from p1_utils import *
'''
  File name: p1_main.py
  Author:
  Date:
'''
# Other functions such as for 3D visualization methods should be in the file p1_main.py

'''

  x axis is weight, y axis is bias, z axis is function output
'''


def Sigmoid_relu_plot(weight, bias, output_sig, output_relu):
    fig = plt.figure()
    fig2 = plt.figure()
    ax = fig.gca(projection='3d')
    ax2 = fig2.gca(projection='3d')
    # plot sigmoid
    surf = ax.plot_surface(weight, bias, output_sig)
    plt.show()
    # plot relu
    surf2 = ax2.plot_surface(weight, bias, output_relu)
    plt.show()
    return


'''
  x axis is weight, y axis is bias, z axis is loss output
'''


def L2_loss_plot(weight, bias, output):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(weight, bias, output)
    plt.show()
    return


'''
  x axis is weight, y axis is bias, z axis is loss output
'''


def cross_entropy_loss_plot(weight, bias, output):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(weight, bias, output)
    plt.show()
    return


# y = W * x + b, W = weight x = 1, gt = 0.5,
weights = np.arange(-1, 1, 0.01)
biases = np.arange(-1, 1, 0.01)
X, Y = np.meshgrid(weights, biases)
x = np.ones([1, 1])
x_input = X * x + Y
#L2_loss_plot(X, Y, x_input)
ground_truth = 0.5  # np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

y_sig = Sigmoid(x_input)
# print(x_input)
# print(y_sig)
y_relu = Relu(x_input)
loss_l2_sig = L2_loss(y_sig, ground_truth)
#np.meshgrid(loss_l2_sig, loss_l2_sig)
print(loss_l2_sig)
loss_ce_sig = Cross_entropy_loss(y_sig, ground_truth)
print(loss_ce_sig)

#Sigmoid_relu_plot(X, Y, y_sig, y_relu)
L2_loss_plot(X, Y, loss_l2_sig)
# print(X.shape)
# print(loss_l2_sig.shape)
cross_entropy_loss_plot(X, Y, loss_ce_sig)
