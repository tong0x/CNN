import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
  File name: p1_main.py
  Author:
  Date:
'''
# Other functions such as for 3D visualization methods should be in the file p1_main.py

'''
  x axis is weight, y axis is bias, z axis is function output
'''


def Sigmoid_relu_plot(weight, bias, output):
    fig = plt.figure()
    ax = fig.plot_surface(X, Y, Z, *args, **kwargs)
    return


'''
  x axis is weight, y axis is bias, z axis is loss output
'''


def L2_loss_plot(weight, bias, output):
    fig = plt.figure()
    return


'''
  x axis is weight, y axis is bias, z axis is loss output
'''


def cross_entropy_loss_plot(weight, bias, output):
    fig = plt.figure()
    return
