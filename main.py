# importing the required module
from operator import xor
import matplotlib.pyplot as plt
from tensorflow.python.ops import math_ops
import numpy as np
from numpy import *
from numpy import random
import json

from tensorflow.python.platform.tf_logging import error
import tensorflow.compat.v1 as tf
import pandas as pd
import time
import signal
import multiprocessing
import pickle
import os
import sys
import tensorflow.compat.v1.keras as keras
import tensorflow.compat.v1.keras.backend as K

path = "/home/radmir/dev/HoloDecoder"

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
    for i in range(0, N):
        FInv[i] = 1 - F[i]
        for j in range(0, N):
            k = j - i
            L = np.hypot(l, k)
            Rcos = np.cos((L / w - np.fix(L / w)) / w * 2 * np.pi)
            R = F[i] * Rcos
            Sl[j] = Sl[j] + R
    return Sl

def initialize_random():
    seed = int.from_bytes(os.urandom(4), sys.byteorder)
    random.seed(seed)

def add_random_noise(message, obj_size, error_percent):
    initialize_random()

    message_with_errors = array(message).view()
    errors = np.fix(obj_size * error_percent).astype(np.int)
    indexes = random.choice(obj_size, size=(errors), replace=False)

    for index in indexes:
        message_with_errors[index] = 1 if message_with_errors[index] == 0 else 0
    return message_with_errors

def normalize(list):
    norm_coef = (1 / max(abs(max(list)), abs(min(list))))
    return (np.array(list) * norm_coef).tolist()

def generate_set(set_size, range_of_excluded_errors, size, name):
    initialize_random()

    set = [0] * set_size
    values = random.bytes(set_size)
    for i in range(0, set_size):
        print(i)
        value = values[i]
        
        amount_of_errors = range_of_excluded_errors[0]
        while amount_of_errors <= range_of_excluded_errors[1] and amount_of_errors >= range_of_excluded_errors[0]:
            amount_of_errors = random.randint(0, 100) / 100

        message = code(value, size)
        message_with_errors = add_random_noise(message, size, amount_of_errors);
        decoded_message_with_errors = decode(message_with_errors, size)

        result = {}
        result['value'] = value / 255
        result['discrete'] = message_with_errors
        result['discrete_without_errors'] = message
        result['decoded'] = normalize(decoded_message_with_errors)
        result['decoded_non_normalized'] = decoded_message_with_errors
        result['amount_of_errors'] = amount_of_errors

        set[i] = result

    with open(f'{path}/data/{name}', 'w+b') as f:
        pickle.dump(set, f)

def load(name, x_entries="decoded"):
    data = []
    with open(f'{path}/data/{name}', 'rb') as f:
        data = pickle.load(f)

    xs = []
    ys = []
    errors = []
    for entry in data:
        xs.append(entry[x_entries])
        ys.append([entry['value']])
        errors.append([entry['amount_of_errors']])
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    return (xs, ys, errors)

def save_metric(history_callback, model_name, metric_name):
    metric = history_callback.history[metric_name]
    arr = np.array(metric)
    np.savetxt(f'{path}/models/{model_name}/{metric_name}_history.txt', arr, delimiter=",")

def train_model(set, model_name, epoch_count, k, rate, loss_func="mean_absolute_error", droppout=0):
    xs, ys, _ = set
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(100 * k, activation=tf.nn.sigmoid))
    if droppout != 0:
        model.add(tf.keras.layers.Dropout(droppout))
    model.add(tf.keras.layers.Dense(50 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    if loss_func == "precise_custom_loss":
        model.compile(optimizer=opt, loss=precise_custom_loss, metrics=[percent_validation])
    else:
        model.compile(optimizer=opt, loss=loss_func, metrics=[percent_validation])
    history_callback = model.fit(xs, ys, epochs=epoch_count, batch_size=64, validation_split = 0.2)

    tf.keras.models.save_model(model, path + '/models/' + model_name)
    save_metric(history_callback, model_name, "loss")
    save_metric(history_callback, model_name, "val_loss")
    save_metric(history_callback, model_name, "percent_validation")
    save_metric(history_callback, model_name, "val_percent_validation")

def percent_validation(y_actual, y_pred):
    # Input is array of rows?
    y_actual = tf.transpose(y_actual)
    y_pred = tf.transpose(y_pred)

    threshold = tf.constant(1 / 256 / 2)
    y_actual_abs = tf.abs(y_actual);
    y_pred_abs = tf.abs(y_pred);
    sub = tf.subtract(y_actual_abs, y_pred_abs)
    sub_abs = tf.abs(sub)
    errors = tf.subtract(sub_abs, threshold)
    errors = tf.sign(errors) # positive = wrong classification
    errors = tf.add(errors, tf.constant(1.0))
    errors = tf.multiply(errors, tf.constant(0.5)) # 1 = wrong
    errors_count = tf.reduce_sum(errors, axis=-1)
    # print(errors_count)

    tmp = tf.add(y_actual_abs, tf.constant(10000.0))
    ons = tf.divide(tmp, tmp)
    total_count = tf.reduce_sum(ons, axis=-1)
    errors_percent = tf.multiply(tf.divide(errors_count, total_count), tf.constant(100.0))
    # errors_percent = tf.divide(errors_count, total_count)
    return errors_percent

def precise_custom_loss(y_actual, y_pred):
    # Input is array of rows?
    y_actual = tf.transpose(y_actual)
    y_pred = tf.transpose(y_pred)    

    x = tf.subtract(y_actual, y_pred)

    return tf.reduce_mean(ls(x), axis=-1)

def ls(x):
    # 0.8
    threshold = 1 / 256 / 2
    x_coef = 2 / threshold
    a = tf.constant(-0.001)
    # a = tf.constant(1.0)

    nom = tf.abs(tf.subtract(tf.constant(2.0), a))
    mul1 = tf.divide(nom, a)
    x = tf.multiply(x, tf.constant(x_coef))
    x = tf.multiply(x, x)
    mul2 = tf.subtract(tf.pow(tf.add(tf.divide(tf.multiply(x, x), nom), tf.constant(1.0)), tf.div(a, tf.constant(2.0))), tf.constant(1.0))

    return tf.multiply(mul1, mul2)

def evaluate_model(set_name, model_name):
    xs, ys, _ = load(set_name)
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    model = tf.keras.models.load_model(path + '/models/' + model_name)
    result = model.evaluate(x=xs, y=ys, use_multiprocessing=True)

figure_idx = 0

def show_losses(models):
    global figure_idx
    figure_idx = figure_idx + 1
    plt.figure(figure_idx)
    fig, axs = plt.subplots(len(models))
    for i, model_name in enumerate(models):
        loss = np.loadtxt(f"{path}/models/{model_name}/loss_history.txt", delimiter=",")
        val_loss = np.loadtxt(f"{path}/models/{model_name}/val_loss_history.txt", delimiter=",")
        percent_validation = np.loadtxt(f"{path}/models/{model_name}/percent_validation_history.txt", delimiter=",")
        val_percent_validation = np.loadtxt(f"{path}/models/{model_name}/val_percent_validation_history.txt", delimiter=",")
        x = range(0, loss.size)
        axs[i].set_title(f'{model_name} loss.min: {round(loss.min(), 3)}/{round(val_loss.min(), 3)} percent_validation.min: {round(percent_validation.min(), 1)}/{round(val_percent_validation.min(), 1)}')
        axs[i].plot(x, loss, color="blue")
        axs[i].plot(x, val_loss, color="red")
        axs[i].legend(['loss', 'val_loss'])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel('Metric')
    fig.tight_layout()

def run_in_new_process(func, a):
    if __name__ == '__main__':
        p1 = multiprocessing.Process(target=func, args=a)
        p1.start()
        return p1

def print_data(set, idx, eval_model):
    model = tf.keras.models.load_model(f'{path}/models/{eval_model}')
    global figure_idx
    figure_idx = figure_idx + 1    
    plt.figure(figure_idx)
    fig, axs = plt.subplots(len(idx))
    for i, set_index in enumerate(idx):
        (xs, ys, errors) = set
        point_xs = xs[set_index]
        point_ys = ys[set_index]
        l = []
        l.append(point_xs)
        l = np.array(l)
        k = []
        k.append(point_ys)
        k = np.array(k)
        result = model.evaluate(x=l, y=k)
        predicted = round((model.predict(x=l)[0][0] * 255))

        x = range(0, point_xs.size)
        axs[i].set_title(f'index: {set_index} error: {round(errors[set_index][0], 2)} value: {point_ys[0] * 255} ({predicted}) loss({eval_model}): {round(result[0], 3)}' )
        axs[i].bar(x, point_xs, color="blue")
        axs[i].legend(['x'])
        axs[i].set_xlabel('Sample')
        axs[i].set_ylabel('Value')
    fig.tight_layout()
    
    (xs, ys, errors) = set
    model.evaluate(x=xs, y=ys)

def show_coded_and_decoded():
    x = range(0, 256)
    codes   = [code(255, 256), code(0, 256), code(100, 256)]
    decoded = [decode(codes[0], 256), decode(codes[1], 256), decode(codes[2], 256)]
    fig, axs = plt.subplots(6)
    axs[0].bar(x, codes[0])
    axs[1].bar(x, decoded[0])
    axs[2].bar(x, codes[1])
    axs[3].bar(x, decoded[1])
    axs[4].bar(x, codes[2])
    axs[5].bar(x, decoded[2])
    fig.tight_layout()
    plt.show()

def concat_sets(sets, resulting_set):
    data = []
    for set in sets:
        with open(f'{path}/data/{set}', 'rb') as f:
            data.append(pickle.load(f))
    data = [item for sublist in data for item in sublist]
    with open(f'{path}/data/{resulting_set}', 'w+b') as f:
        pickle.dump(data, f)        

def generate_sets(index, size, entries_count, range_of_excluded_errors, workers):
    processes = []
    for i in range(0, workers):
        processes.append(run_in_new_process(generate_set, (entries_count, range_of_excluded_errors, size, f"v2_0_1_set_{size}_{entries_count}_[{range_of_excluded_errors[0]}:{range_of_excluded_errors[1]}]_{index}.pickle")))
        index += 1
    for p in processes:
        p.join()   

def show_results(set_name, offset, model):
    set = load(set_name)
    print_data(set, [i + offset for i in [1, 2, 3, 4]], model)
    plt.show()

def show_losses_macro(gen, loss_funcs, droppout, epochs):
    i = 1
    for loss_func in loss_funcs: 
        show_losses([
            f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model{i}_1", 
            f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model{i}_2", 
            f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model{i}_3"
        ])
        i = i + 1
    plt.show()


# def ls(x):
#     # 0.8
#     threshold = 1 / 256 / 2
#     x_coef = 2 / threshold
#     a = tf.constant(1.0)

#     nom = tf.abs(tf.subtract(tf.constant(2.0), a))
#     mul1 = tf.divide(nom, a)
#     x = tf.multiply(x, tf.constant(x_coef))
#     x = tf.multiply(x, x)
#     mul2 = tf.subtract(tf.pow(tf.add(tf.divide(tf.multiply(x, x), nom), tf.constant(1.0)), tf.div(a, tf.constant(2.0))), tf.constant(1.0))

#     return tf.multiply(mul1, mul2)

# xs = tf.range(start=-2, limit=2, delta=0.0001, dtype=tf.float32, name='range')
# res = ls(xs).numpy()
# x = range(0, len(res))
# plt.plot(x, res, color="black")
# plt.show()

# show_results("v2_0_1_set_256_10000_[0.4:0.6]_27.pickle", 0, "g3_e2000_lmean_squared_error_d0.2_model3_1")
# show_losses_macro(gen=5, loss_funcs=["mean_absolute_error", "mean_squared_error"], droppout=0, epochs=1000)
# show_losses_macro(gen=3, loss_func="mean_squared_error", droppout=0.2, epochs=2000)

# processes = []
# set = load("v2_0_1_set_256_10000_13.pickle")
# epochs = 1000
# processes.append(run_in_new_process(train_model, (set, "g0_model1_1", epochs, 0.5, 0.001)))
# for p in processes:
#     p.join()

# show_losses(["g0_model1_1", "g0_model1_1"])
# plt.show()
    
# x = range(0, 256)
# message = code(100, 256)
# w_err_0 = add_random_noise(message, 256, 0)
# w_err_25 = add_random_noise(message, 256, 0.25)
# w_err_50 = add_random_noise(message, 256, 0.5)
# w_err_75 = add_random_noise(message, 256, 0.75)
# w_err_100 = add_random_noise(message, 256, 1)
# # fig, axs = plt.subplots(5)
# # axs[0].bar(x, w_err_0, color="black")
# # axs[1].bar(x, w_err_25, color="black")
# # axs[2].bar(x, w_err_50, color="black")
# # axs[3].bar(x, w_err_75, color="black")
# # axs[4].bar(x, w_err_100, color="black")

# d_0 = decode(w_err_0, 256)
# d_25 = decode(w_err_25, 256)
# d_50 = decode(w_err_50, 256)
# d_75 = decode(w_err_75, 256)
# d_100 = decode(w_err_100, 256)
# fig, axs = plt.subplots(5)
# # axs[0].bar(x, d_0, color="black")
# # axs[1].bar(x, d_25, color="black")
# # axs[2].bar(x, d_50, color="black")
# # axs[3].bar(x, d_75, color="black")
# # axs[4].bar(x, d_100, color="black")
# axs[0].bar(x, normalize(d_0), color="black")
# axs[1].bar(x, normalize(d_25), color="black")
# axs[2].bar(x, normalize(d_50), color="black")
# axs[3].bar(x, normalize(d_75), color="black")
# axs[4].bar(x, normalize(d_100), color="black")
# plt.show()



# x = range(0, 256)
# (xs, ys, _)  = load("v2_0_1_set_256_10_[0.1:0.9]_7.pickle")
# m1 = xs[0]
# (xs, ys, _)  = load("v2_0_1_set_256_10_[0.1:0.9]_8.pickle")
# m2 = xs[0]
# (xs, ys, _)  = load("v2_0_1_set_256_10_[0.1:0.9]_9.pickle")
# m3 = xs[0]
# (xs, ys, _)  = load("v2_0_1_set_256_10_[0.1:0.9]_10.pickle")
# m4 = xs[0]
# fig, axs = plt.subplots(4)
# axs[0].bar(x, m1, color="black")
# axs[1].bar(x, m2, color="black")
# axs[2].bar(x, m3, color="black")
# axs[3].bar(x, m4, color="black")
# plt.show()

# concat_sets([
#     "v2_0_1_set_256_60000_[0.4:0.6]_7-12.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_13.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_14.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_15.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_16.pickle",

#     "v2_0_1_set_256_10000_[0.4:0.6]_17.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_18.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_19.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_20.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_21.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_22.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_23.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_24.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_25.pickle",
#     "v2_0_1_set_256_10000_[0.4:0.6]_26.pickle",
# ],
# "v2_0_1_set_256_200000_[0.4:0.6]_7-26.pickle")

# y_actual = tf.constant(  [0, 0.1, 0.2,  0.3,    1,   1,   1     , -1, -2])
# y_expected = tf.constant([0, 0.1, 0.25, 0.3001, 1.3, 0.9, 0.9999, -1, -2.3])
# print(percent_validation(y_actual, y_expected))

# y_actual = tf.constant(  [[0], [0.1], [0.2],  [0.3],    [1],   [1],   [1]])
# y_expected = tf.constant([[0], [0.1], [0.25], [0.3001], [1.3], [0.9], [0.9999]])
# print(percent_validation(y_actual, y_expected))

# y_actual = tf.transpose(y_actual)
# y_expected = tf.transpose(y_expected)
# print(percent_validation(y_actual, y_expected))

processes = []
set = load("v2_0_1_set_256_200000_[0.4:0.6]_7-26.pickle")
loss_func = "precise_custom_loss"
droppout = 0
epochs = 1000
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model1_1", epochs, 1, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model1_2", epochs, 1, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model1_3", epochs, 1, 0.012, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model1_4", epochs, 0.5, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model1_5", epochs, 0.5, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model1_6", epochs, 0.5, 0.012, loss_func, droppout)))

droppout = 0.2
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model2_1", epochs, 1, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model2_2", epochs, 1, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model2_3", epochs, 1, 0.012, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model2_4", epochs, 0.5, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model2_5", epochs, 0.5, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model2_6", epochs, 0.5, 0.012, loss_func, droppout)))
for p in processes:
    p.join()

processes = []
set = load("v2_0_1_set_256_200000_[0.4:0.6]_7-26.pickle")
loss_func = "mean_absolute_error"
droppout = 0
epochs = 1000
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model3_1", epochs, 1, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model3_2", epochs, 1, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model3_3", epochs, 1, 0.012, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model3_4", epochs, 0.5, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model3_5", epochs, 0.5, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model3_6", epochs, 0.5, 0.012, loss_func, droppout)))

loss_func = "mean_squared_error"
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model4_1", epochs, 1, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model4_2", epochs, 1, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model4_3", epochs, 1, 0.012, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model4_4", epochs, 0.5, 0.003, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model4_5", epochs, 0.5, 0.006, loss_func, droppout)))
processes.append(run_in_new_process(train_model, (set, f"g5_e{epochs}_l{loss_func}_d{droppout}_model4_6", epochs, 0.5, 0.012, loss_func, droppout)))
for p in processes:
    p.join()

# generate_sets(index=320, size=256, entries_count=10000, range_of_excluded_errors=[0, 1], workers=10)

# processes = []
# gen = 4
# set = load("v2_0_1_set_256_60000_[0.4:0.6]_7-12.pickle")
# loss_func = "precise_custom_loss"
# droppout = 0.2
# epochs = 100
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model1_1", epochs, 0.5, 0.003, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model1_2", epochs, 0.5, 0.006, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model1_3", epochs, 0.5, 0.0012, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model2_1", epochs, 1, 0.003, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model2_2", epochs, 1, 0.006, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model2_3", epochs, 1, 0.0012, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model3_1", epochs, 2, 0.003, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model3_2", epochs, 2, 0.006, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g{gen}_e{epochs}_l{loss_func}_d{droppout}_model3_3", epochs, 2, 0.0012, loss_func, droppout)))
# for p in processes:
#     p.join()

# generate_sets(index=170, size=256, entries_count=10000, range_of_excluded_errors=[0.3, 0.7], workers=10)
# generate_sets(index=180, size=256, entries_count=10000, range_of_excluded_errors=[0.3, 0.7], workers=10)
# generate_sets(index=190, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=200, size=256, entries_count=10000, range_of_excluded_errors=[0.3, 0.7], workers=10)
# generate_sets(index=210, size=256, entries_count=10000, range_of_excluded_errors=[0.3, 0.7], workers=10)
# generate_sets(index=220, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=230, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=240, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=250, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=260, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=270, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=280, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=290, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=300, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)
# generate_sets(index=310, size=256, entries_count=10000, range_of_excluded_errors=[0.4, 0.6], workers=10)

# processes = []
# set = load("v2_0_1_set_256_60000_[0.4:0.6]_7-12.pickle")
# loss_func = "mean_absolute_error"
# droppout = 0.2
# epochs = 2000
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model1_1", epochs, 0.5, 0.003, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model1_2", epochs, 0.5, 0.006, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model1_3", epochs, 0.5, 0.0012, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model2_1", epochs, 1, 0.003, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model2_2", epochs, 1, 0.006, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model2_3", epochs, 1, 0.0012, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model3_1", epochs, 2, 0.003, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model3_2", epochs, 2, 0.006, loss_func, droppout)))
# processes.append(run_in_new_process(train_model, (set, f"g3_e{epochs}_l{loss_func}_d{droppout}_model3_3", epochs, 2, 0.0012, loss_func, droppout)))
# for p in processes:
#     p.join()

# generate_sets(index=40, size=256, entries_count=10000, range_of_excluded_errors=[0.35, 0.65], workers=10)
# generate_sets(index=50, size=256, entries_count=10000, range_of_excluded_errors=[0.35, 0.65], workers=10)
# generate_sets(index=60, size=256, entries_count=10000, range_of_excluded_errors=[0.40, 0.60], workers=10)
# generate_sets(index=70, size=256, entries_count=10000, range_of_excluded_errors=[0.40, 0.60], workers=10)
# generate_sets(index=80, size=256, entries_count=10000, range_of_excluded_errors=[0.30, 0.70], workers=10)
# generate_sets(index=90, size=256, entries_count=10000, range_of_excluded_errors=[0.30, 0.70], workers=10)