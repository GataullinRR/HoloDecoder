# importing the required module
import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from numpy import random
import json
import tensorflow.compat.v1 as tf
import pandas as pd
import time
import signal
import multiprocessing

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
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    return (xs, ys, errors)

def save_metric(history_callback, model_name, metric_name):
    metric = history_callback.history[metric_name]
    arr = np.array(metric)
    np.savetxt('C:/HoloDecoder/models/' + model_name + "/" + metric_name + "_history.txt", arr, delimiter=",")

def train_model(set_name, model_name, k, rate):
    xs, ys, _ = load(set_name)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    train_set_size = int(fix(ys.size * 0.8))
    xs_train = xs[1:train_set_size];
    ys_train = ys[1:train_set_size];
    xs_test = xs[train_set_size:ys.size];
    ys_test = ys[train_set_size:ys.size];

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(30 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(10 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss='mean_absolute_error', metrics=["mean_absolute_error"])
    history_callback = model.fit(xs_train, ys_train, epochs=5000, validation_data=(xs_test, ys_test), batch_size=64)

    tf.keras.models.save_model(model, 'C:/HoloDecoder/models/' + model_name)
    save_metric(history_callback, model_name, "loss")
    save_metric(history_callback, model_name, "val_loss")

def evaluate_model(set_name, model_name):
    xs, ys, _ = load(set_name)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    model = tf.keras.models.load_model('C:/dev/HoloDecoder/models/' + model_name)
    result = model.evaluate(x=xs, y=ys, use_multiprocessing=True)

generate_set(10000, 256, "set_256_5000_5.json")
generate_set(10000, 256, "set_256_5000_6.json")

# train_model("set_256_20000_1+2.json", "g2_model1_1", 0.5, 0.001)
# train_model("set_256_20000_1+2.json", "g2_model1_2", 0.5, 0.003)
# train_model("set_256_20000_1+2.json", "g2_model1_3", 0.5, 0.006)
# train_model("set_256_20000_1+2.json", "g2_model2_1", 1, 0.001)
# train_model("set_256_20000_1+2.json", "g2_model2_2", 1, 0.003)
# train_model("set_256_20000_1+2.json", "g2_model2_3", 1, 0.006)
# train_model("set_256_20000_1+2.json", "g2_model3_1", 2, 0.001)
# train_model("set_256_20000_1+2.json", "g2_model3_2", 2, 0.003)
# train_model("set_256_20000_1+2.json", "g2_model3_3", 2, 0.006)