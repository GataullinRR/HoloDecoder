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
import pickle
import os
import sys

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

def generate_set(set_size, size, name):
    initialize_random()

    set = [0] * set_size
    values = random.bytes(set_size)
    print(values)
    for i in range(0, set_size):
        print(i)
        value = values[i]
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

def load(name):
    data = []
    with open(f'{path}/data/{name}', 'rb') as f:
        data = pickle.load(f)

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
    np.savetxt(f'{path}/models/{model_name}/{metric_name}_history.txt', arr, delimiter=",")

def train_model(set, model_name, epoch_count, k, rate):
    xs, ys, _ = set
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(30 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(10 * k, activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    opt = tf.keras.optimizers.Adam(learning_rate=rate)
    model.compile(optimizer=opt, loss='mean_absolute_error', metrics=["mean_absolute_error"])
    history_callback = model.fit(xs, ys, epochs=epoch_count, batch_size=64, validation_split = 0.2)

    tf.keras.models.save_model(model, path + '/models/' + model_name)
    save_metric(history_callback, model_name, "loss")
    save_metric(history_callback, model_name, "val_loss")

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
        x = range(0, loss.size)
        axs[i].set_title(f'{model_name} loss.min: {round(loss.min(), 3)} val_loss.min: {round(val_loss.min(), 3)}')
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

def generate_sets(index, size, entries_count, workers):
    processes = []
    for i in range(0, workers):
        processes.append(run_in_new_process(generate_set, (entries_count, size, f"v2_0_1_set_{size}_{entries_count}_{index}.pickle")))
        index += 1
    for p in processes:
        p.join()   

def show_results(set_name, offset, model):
    set = load(set_name)
    print_data(set, [i + offset for i in [1, 2, 3, 4]], model)
    plt.show()

def show_losses_macro():
    show_losses(["g3_model1_1", "g3_model2_1", "g3_model3_1"])
    show_losses(["g3_model1_2", "g3_model2_2", "g3_model3_2"])
    show_losses(["g3_model1_3", "g3_model2_3", "g3_model3_3"])
    show_losses(["g3_model1_4", "g3_model2_4", "g3_model3_4"])
    plt.show()

# show_losses_macro();

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

# generate_sets(index=35, size=256, entries_count=10, workers=4)

# x = range(0, 256)
# (xs, ys, _)  = load("v2_0_1_set_256_10_35.pickle")
# m1 = xs[0]
# (xs, ys, _)  = load("v2_0_1_set_256_10_36.pickle")
# m2 = xs[0]
# (xs, ys, _)  = load("v2_0_1_set_256_10_37.pickle")
# m3 = xs[0]
# (xs, ys, _)  = load("v2_0_1_set_256_10_38.pickle")
# m4 = xs[0]
# fig, axs = plt.subplots(4)
# axs[0].bar(x, m1, color="black")
# axs[1].bar(x, m2, color="black")
# axs[2].bar(x, m3, color="black")
# axs[3].bar(x, m4, color="black")
# plt.show()

# show_losses_macro()
# show_results("v2_0_1_set_256_60000_13-18.pickle", 000, "g3_model3_1")
# show_results("v2_0_1_set_256_10000_20.pickle", 000, "g3_model3_1")
# show_results("v2_0_1_set_256_10000_18.pickle", "g3_model3_3")
# show_results("v2_set_256_100_14.pickle", "g3_model3_1")
# show_results("v2_set_256_100_15.pickle", "g3_model3_1")

# concat_sets([
#     "v2_set_256_10000_1.pickle",
#     "v2_set_256_10000_2.pickle",
#     "v2_set_256_10000_3.pickle",
#     "v2_set_256_10000_4.pickle",
#     "v2_set_256_10000_5.pickle",
#     "v2_set_256_10000_6.pickle"
# ],
# "v2_set_256_60000_1-6.pickle")

# generate_sets(index=20, size=256, entries_count=10000, workers=9)

# processes = []
# set = load("v2_0_1_set_256_60000_13-18.pickle")
# epochs = 3000
# processes.append(run_in_new_process(train_model, (set, "g3_model1_1", epochs, 0.5, 0.001)))
# processes.append(run_in_new_process(train_model, (set, "g3_model1_2", epochs, 0.5, 0.003)))
# processes.append(run_in_new_process(train_model, (set, "g3_model1_3", epochs, 0.5, 0.006)))
# processes.append(run_in_new_process(train_model, (set, "g3_model1_4", epochs, 0.5, 0.012)))
# processes.append(run_in_new_process(train_model, (set, "g3_model2_1", epochs, 1, 0.001)))
# processes.append(run_in_new_process(train_model, (set, "g3_model2_2", epochs, 1, 0.003)))
# processes.append(run_in_new_process(train_model, (set, "g3_model2_3", epochs, 1, 0.006)))
# processes.append(run_in_new_process(train_model, (set, "g3_model2_4", epochs, 1, 0.012)))
# processes.append(run_in_new_process(train_model, (set, "g3_model3_1", epochs, 2, 0.001)))
# processes.append(run_in_new_process(train_model, (set, "g3_model3_2", epochs, 2, 0.003)))
# processes.append(run_in_new_process(train_model, (set, "g3_model3_3", epochs, 2, 0.006)))
# processes.append(run_in_new_process(train_model, (set, "g3_model3_4", epochs, 2, 0.012)))
# for p in processes:
#     p.join()

# generate_sets(index=29, size=256, entries_count=10000, workers=6)