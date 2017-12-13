'''
  File name: p2_train.py
  Author:
  Date:
'''


import PyNet as net
import numpy as np
from p2_utils import *
import matplotlib as mp
import matplotlib.pyplot as plt
# import plotly.plotly as py

'''
  network architecture construction
  - Stack layers in order based on the architecture of your network
'''

#


def randomShuffle(data_set, label_set):
  temp = np.arange(data_set.shape[0])
  np.random.shuffle(temp)
  data_set_cur = data_set[temp]
  label_set_cur = label_set[temp]

  # perm = np.random.permutation(data_set.shape[0])
  # data_set_cur = data_set[perm,:]
  # label_set_cur = label_set[perm]
  return data_set_cur, label_set_cur


input_images = "p21_random_imgs.npy"
input_labels = "p21_random_labs.npy"


layer_list = [
    # net.Conv2d(16, 7, padding=0, stride=1),
    # net.BatchNorm2D(),
    # net.Relu(),
    # net.Conv2d(8, 7, padding=0, stride=1),
    # net.BatchNorm2D(),
    # net.Relu(),
    # net.Flatten(),
    net.Flatten(),
    net.Linear(4 * 4, 4, bias=True),
    net.Sigmoid(),
    net.Linear(4, 1, bias=True),
    net.Sigmoid(),
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
my_model.set_input_channel(16)

'''
  Main training process
  - train N epochs, each epoch contains M steps, each step feed a batch-sized data for training,
    that is, total number of data = M * batch_size, each epoch need to traverse all data.
'''

# obtain data
[data_set, label_set] = np.load(input_images), np.load(input_labels)

plot_list = []
train_iterations = []
accuracy_list = []
quit = False

for i in range(1000):
  train_iterations.append(i)
  '''
    random shuffle data
  '''
  data_set_cur, label_set_cur = randomShuffle(data_set, label_set)  # design function by yourself

  step = 1  # step is a int number
  # step is the iteration needed to run through all the data, step = data_num/batch_size.
  # You can drop the last batch if data_num is not divisible by the batch_size
  for j in range(step):
    # Get mini batch for this training step
    [data_bt, label_bt] = data_set_cur, label_set_cur

    # feedward data and label to the model
    loss, pred = my_model.forward(data_bt, label=label_bt)
    plot_list.append(loss)
    rounded_pred = np.zeros(pred.shape)

    rounded_pred[pred > .5] = 1
    rounded_pred[pred <= .5] = 0
    new_pred = np.reshape(rounded_pred, (-1, 1))
    accuracy = 1 - float(np.sum(np.logical_xor(new_pred, np.reshape(label_bt, (-1, 1))))) / float(label_bt.size)
    accuracy = 1 - np.mean(np.logical_xor(new_pred, np.reshape(label_bt, (-1, 1))))
    #accuracy = np.mean(label_bt == pred)

    accuracy_list.append(accuracy)

    # backward loss
    my_model.backward(loss)

    # update parameters in model
    my_model.update_param()
    if(accuracy == 1.0):
      quit = True
    else:
      print accuracy

    if quit == True:
      # print "yay"
      break


#plt.plot(train_iterations, plot_list)
plt.plot(train_iterations, accuracy_list)
#plt.ylim([0, 1])
# plt.show()
plt.savefig('2.1.1b.png')
