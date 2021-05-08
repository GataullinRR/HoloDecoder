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

def create_train_model(size, hidden_nodes, num_iters, loss_plot):
    # Reset the graph
    tf.reset_default_graph()
    tf.disable_v2_behavior()

    data = []
    with open('C:/dev/holodecoder_training_set_1.json') as f:
        data = json.load(f)
    
    xs = []
    ys = []

    xmin = 0
    xmax = 0
    for entry in data:
        curr_min = min(entry['decoded'])
        curr_max = max(entry['decoded'])
        xmin = curr_min if curr_min < xmin else xmin
        xmax = curr_max if curr_min > xmax else xmax

    for entry in data:
        fixed = list(map(lambda x: (x + abs(xmin)) / (abs(xmin) + abs(xmax)), entry['decoded']))
        xs.append([fixed])
        ys.append([entry['value'] / 256.0])
    
    # Xtrain = frame
    # #Xtest = data.head(200).drop(columns=['amount_of_errors', 'value']).explode('decoded').transpose()
    # ytrain = pd.get_dummies('value')
    # # ytest = pd.get_dummies(data.head(200).value)
    
    # print(frame)
    # print(Xtrain)
    # print(ytrain)

    # Placeholders for input and output data
    X = tf.placeholder(shape=(1000, size), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(1000, 1), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(size, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 1), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    #deltas = tf.square(y_est - y)
    deltas = tf.abs(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    pdxs = pd.DataFrame(np.concatenate(xs))
    pdys = pd.DataFrame(np.concatenate(ys)) 

    print(pdxs)
    print(pdys)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: pdxs, y: pdys})
        loss = sess.run(loss, feed_dict={X: pdxs.values, y: pdys.values})
        loss_plot[hidden_nodes].append(loss)
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)
        
    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2

size = 256

num_iters = 100
num_hidden_nodes = [5, 10, 20]
loss_plot = {5: [], 10: [], 20: []}
weights1 = {5: None, 10: None, 20: None}
weights2 = {5: None, 10: None, 20: None}

plt.figure(figsize=(12,8))
for hidden_nodes in num_hidden_nodes:
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(size, hidden_nodes, num_iters, loss_plot)
    plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 4-%d-3" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
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