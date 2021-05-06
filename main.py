# importing the required module
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from numpy import random
# import tensorflow as tf

def code(value, obj_size):
    obj_size = 256
    dist_to_hologram = obj_size + 0.5
    wave_len = 1
    value = 100

    prj = [0] * obj_size
    prj[value] = 1

    point_hologram = [0] * obj_size
    hologram = [0] * obj_size
    for i in range(0, obj_size):
        j = i - value
        hypot = np.hypot(dist_to_hologram, j)
        hologram[i] = 1 * np.cos((hypot / wave_len - np.fix(hypot / wave_len)) / wave_len * 2 * np.pi)
        point_hologram[i] = 1 if hologram[i] > 0 else 0
    return point_hologram

def decode(msg, obj_size):
    dist_to_hologram = obj_size + 0.5
    wave_len = 1
    value = 100

    w = wave_len
    l = dist_to_hologram
    N = obj_size
    F = msg
    FInv = [0] * N
    Sl = [0] * N
    SInvl = [0] * N
    for i in range(0, N):
        FInv[i] = 1 - F[i]
        for j in range(0, N):
            k = j - i
            L = np.hypot(l, k)
            Rcos = np.cos((L / w - np.fix(L / w)) / w * 2 * np.pi)
            R = F[i] * Rcos
            Sl[j] = Sl[j] + R
            RInv = FInv[i] * Rcos
            SInvl[j] = SInvl[j] + RInv
    return Sl

def add_random_noise(message, obj_size, error_percent):
    message_with_errors = array(message).view()
    errors = np.fix(obj_size * error_percent).astype(np.int)
    indexes = random.choice(obj_size, size=(errors), replace=False)
    print(indexes)

    for index in indexes:
        message_with_errors[index] = 1 if message_with_errors[index] == 0 else 1
    return message_with_errors

size = 256

tries = 3
fig, axs = plt.subplots(tries * 2)
for i in range(0, tries):
    errors = 0.5 * i / tries
    print(errors)
    message = code(25, size)
    message_with_errors = add_random_noise(message, size, errors);
    decoded_message_with_errors = decode(message_with_errors, size)
    
    axs[i * 2].bar(range(0, size), message_with_errors)
    axs[i * 2 + 1].bar(range(0, size), decoded_message_with_errors)
fig.tight_layout()
plt.show()

# tensorflowinputs = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name='inputs')
# label = tf.placeholder(tf.float32, shape=(None, 2), name='labels')

# # First layer
# hid1_size = 128
# w1 = tf.Variable(tf.random_normal([hid1_size, X_train.shape[1]], stddev=0.01), name='w1')
# b1 = tf.Variable(tf.constant(0.1, shape=(hid1_size, 1)), name='b1')
# y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w1, tf.transpose(inputs)), b1)), keep_prob=0.5)

# # Second layer
# hid2_size = 256
# w2 = tf.Variable(tf.random_normal([hid2_size, hid1_size], stddev=0.01), name='w2')
# b2 = tf.Variable(tf.constant(0.1, shape=(hid2_size, 1)), name='b2')
# y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(w2, y1), b2)), keep_prob=0.5)

# # Output layer
# wo = tf.Variable(tf.random_normal([2, hid2_size], stddev=0.01), name='wo')
# bo = tf.Variable(tf.random_normal([2, 1]), name='bo')
# yo = tf.transpose(tf.add(tf.matmul(wo, y2), bo))