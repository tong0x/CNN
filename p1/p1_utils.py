'''
  File name: p1_utils.py
  Author:
  Date:
'''
# Put all the helper functions (e.g. gradient computation) in the file p1_utils.py


# compute the backward gradient of L2 loss with respective to the weight

def Backward_gradient_cross_entropy(x, y, pred):
    return x * (pred - y)


# compute the backward gradient of Cross-entropy loss with respective to the weight

def Backward_gradient_cross_entropy(x, y, pred):
    return x * (pred - y)
