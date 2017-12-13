'''
  File name: p2_train.py
  Author:
  Date:
'''

import PyNet as net
#import myLayers as ml
import numpy as np
from p2_utils import *
import matplotlib as mp
import matplotlib.pyplot as plt
#import plotly.plotly as py


'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''


def randomShuffle(data_set, label_set):
  temp = np.arange(data_set.shape[0])
  np.random.shuffle(temp)
  data_set_cur = data_set[temp]
  label_set_cur = label_set[temp]
  return data_set_cur, label_set_cur


input_images = "p22_line_imgs.npy"
input_labels = "p22_line_labs.npy"
max_epoch_num = 1000

layer_list = [
    net.Conv2d(16, 7, padding=0, stride=1),
    # net.BatchNorm2D(),
    net.Relu(),
    net.Conv2d(8, 7, padding=0, stride=1),
    net.Relu(),
    net.Flatten(),
    net.Linear(input_dim=16 * 8, output_dim=1),
    net.Sigmoid()
    # net.BatchNorm2D(),
    # net.Relu(),
    # net.Flatten(),
    # net.Linear(100 * 128, 10)

]

'''
  Define loss function
'''
loss_layer = net.Binary_cross_entropy_loss()
'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(lr_rate=0.01, weight_decay=5e-4, momentum=0.99)

'''
  Build model
'''
my_model = net.Model(layer_list, loss_layer, optimizer)

'''
  Define the number of input channel and initialize the model
'''
my_model.set_input_channel(1)

'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''

# obtain data
data_set, label_set = np.load(input_images), np.load(input_labels)

plotting_array = []
training_iters = []
accuracy_array = []


for i in range(max_epoch_num):

  training_iters.append(i)
  data_set_cur, label_set_cur = randomShuffle(data_set, label_set)  # design function by yourself

  shouldPrint = False
  if(i == max_epoch_num - 1):
    shouldPrint = True
  step = 1  # step is a int number
  accuracy = 0
  for j in range(step):
    # obtain a mini batch for this step training
    [data_bt, label_bt] = np.reshape(data_set_cur, (-1, 1, 16, 16)), label_set_cur

    # feedward data and label to the model
    loss, pred = my_model.forward(data_bt, np.reshape(label_bt, (-1, 1)))

    pred[pred > .5] = 1
    pred[pred <= .5] = 0
    new_pred = np.reshape(pred, (64,))

    accuracy = 1 - float(np.sum(np.logical_xor(new_pred, label_bt))) / float(label_bt.size)
    accuracy = 1 - np.mean(np.logical_xor(new_pred, np.reshape(label_bt, (-1, 1))))
    plotting_array.append(loss)
    accuracy_array.append(accuracy)
    my_model.backward(loss)

    # update parameters in model
    my_model.update_param()

  if(accuracy == 1.0):
    break
  else:
    print accuracy
    pass

plt.plot(training_iters, plotting_array)
plt.show()
