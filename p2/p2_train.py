'''
  File name: p2_train.py
  Author:
  Date:
'''

import PyNet as net
import numpy as np
from p2_utils import *


'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''

layer_list = [
    #net.Conv2d(16, 7, padding=0, stride=1),
    # net.BatchNorm2D(),
    # net.Relu(),
    #net.Conv2d(8, 7, padding=0, stride=1),
    # net.BatchNorm2D(),
    # net.Relu(),
    # net.Flatten(),
    # net.Linear(100 * 128, 10)
]

'''
  Define loss function
'''
loss_layer = net.L2_loss()
'''
  Define optimizer 
'''
optimizer = net.SGD_Optimizer(lr_rate=0.1, weight_decay=5e-4, momentum=0.99)

'''
  Build model
'''
my_model = net.Model(layer_list, loss_layer, optimizer)

'''
  Define the number of input channel and initialize the model
'''
my_model.set_input_channel(3)

'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''

# obtain data
[data_set, label_set] = loadData(.....)

for i in range(1000):
  '''
    random shuffle data 
  '''
  data_set_cur, label_set_cur = randomShuffle(data_set, label_set)  # design function by yourself

  step = ...  # step is a int number
  # step is the iteration needed to run through all the data, step = data_num/batch_size.
  # You can drop the last batch if data_num is not divisible by the batch_size
  for j in range(step):
    # obtain a mini batch for this step training
    [data_bt, label_bt] = obtainMiniBatch(....)  # design function by yourself

    # feedward data and label to the model
    loss, pred = my_model(data_bt, label_bt)

    # backward loss
    my_model.backward(loss)

    # update parameters in model
    my_model.update_param()
