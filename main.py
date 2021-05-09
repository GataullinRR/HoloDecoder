# importing the required module
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from numpy import random
import json
import tensorflow.compat.v1 as tf
import pandas as pd

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

    for index in indexes:
        message_with_errors[index] = 1 if message_with_errors[index] == 0 else 1
    return message_with_errors

def plot(size):
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

def generate_set(set_size):
    set = [0] * set_size
    values = np.random.bytes(set_size)
    for i in range(0, set_size):
        value = values[i]
        amount_of_errors = random.randint(0, 100) / 100

        message = code(value, size)
        message_with_errors = add_random_noise(message, size, amount_of_errors);
        decoded_message_with_errors = decode(message_with_errors, size)

        result = {}
        result['value'] = value
        result['decoded'] = decoded_message_with_errors
        result['amount_of_errors'] = amount_of_errors
        set[i] = result

    with open('C:/dev/holodecoder_training_set_1.json', 'w+') as f:
        json_str = json.dumps(set, indent=4)
        f.write(json_str)

def create_train_model():
    # Reset the graph
    tf.reset_default_graph()
    tf.disable_v2_behavior()

    data = []
    with open('C:/dev/holodecoder_training_set_1.json') as f:
        data = json.load(f)

    xmin = 0
    xmax = 0
    for entry in data:
        curr_min = min(entry['decoded'])
        curr_max = max(entry['decoded'])
        xmin = curr_min if curr_min < xmin else xmin
        xmax = curr_max if curr_min > xmax else xmax
    
    xs = []
    ys = []
    for entry in data:
        #fixed = list(map(lambda x: (x + abs(xmin)) / (abs(xmin) + abs(xmax)), entry['decoded']))
        #xs.append(fixed)
        xs.append(entry['decoded'])
        # ys.append([entry['value'] / 256.0])
        ys.append([entry['value']])

    xs = tf.keras.utils.normalize(xs)
    ys = tf.keras.utils.normalize(ys)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    xs_train = xs[1:800];
    ys_train = ys[1:800];
    xs_test = xs[800:1000];
    ys_test = ys[800:1000];

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    # As signified by loss = 'mean_squared_error', you are in a regression setting, where accuracy is meaningless (it is meaningful only in classification problems).
    model.compile(optimizer='adam', loss='mse', metrics=["mae"])
    model.fit(xs_train, ys_train, epochs=1000, validation_data=(xs_test, ys_test))

create_train_model()