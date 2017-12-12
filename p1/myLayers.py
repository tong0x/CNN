import numpy as np
'''
  File name: myLayers.py
  Author:
  Date:
'''


'''
  Sigmoid layer
  - Input x: ndarray
  - Output y: nonlinearized result
'''


def Sigmoid(x):
  # TODO
  y = 1 / (1 + np.exp(-x))
  return y


'''
  Relu layer
  - Input x: ndarray 
  - Output y: nonlinearized result
'''


def Relu(x):
  # TODO
  y = np.maximum(0, x)
  return y


'''
  l2 loss layer
  - Input pred: prediction values
  - Input gt: the ground truth 
  - Output loss: averaged loss
'''


def L2_loss(pred, gt):
  # TODO
  loss = ((pred - gt)**2) / 2
  return loss


'''
  Cross entropy loss layer
  - Input pred: prediction values
  - Input gt: the ground truth 
  - Output loss: averaged loss
'''


def Cross_entropy_loss(pred, gt):
  # TODO
  #-(yloga + (1-y)log(1-a)) where a = pred, y = gt
  loss = -(gt * np.log(pred) + (1 - gt) * np.log(1 - pred))

  return loss


list_relu = np.array([[1, 2, -1, -5], [-3, -4, -5, -6]])
relu_y = Sigmoid(list_relu)
pred = 0.3
gt = 0.5
loss = L2_loss(pred, gt)
print(relu_y)
print("L2 LOSS ", loss)
