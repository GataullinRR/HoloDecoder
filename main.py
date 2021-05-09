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

def generate_set(set_size, size, name):
    set = [0] * set_size
    values = np.random.bytes(set_size)
    for i in range(0, set_size):
        value = values[i]
        amount_of_errors = random.randint(0, 100) / 100

        message = code(value, size)
        message_with_errors = add_random_noise(message, size, amount_of_errors);
        decoded_message_with_errors = decode(message_with_errors, size)

        result = {}
        result['value'] = value / 256
        result['decoded'] = tf.keras.utils.normalize(decoded_message_with_errors)[0].tolist()
        result['amount_of_errors'] = amount_of_errors
        set[i] = result

    with open('C:/dev/' + name, 'w+') as f:
        json_str = json.dumps(set, indent=4)
        f.write(json_str)

def load(name):
    data = []
    with open('C:/dev/' + name) as f:
        data = json.load(f)

    xs = []
    ys = []
    errors = []
    for entry in data:
        xs.append(entry['decoded'])
        ys.append([entry['value']])
        errors.append([entry['amount_of_errors']])

    # xs = tf.keras.utils.normalize(xs, axis = 0)
    # ys = tf.keras.utils.normalize(ys, axis = 0)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    return (xs, ys, errors)

def create_train_model():
    # Reset the graph
    tf.reset_default_graph()
    tf.disable_v2_behavior()

    data = []
    with open('C:/dev/holodecoder_training_set_1.json') as f:
        data = json.load(f)

    xs = []
    ys = []
    for entry in data:
        xs.append(entry['decoded'])
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

#create_train_model()

generate_set(10, 256, "set_2.json")
xs, ys, errors = load('set_2.json')
cnt = 5
fig, axs = plt.subplots(cnt)
for i in range(cnt):
    print("Err: " + str(errors[i][0]) + ' - y: ' + str(ys[i][0]))
    axs[i].bar(range(0, xs[i].size), xs[i])
fig.tight_layout()
plt.show()