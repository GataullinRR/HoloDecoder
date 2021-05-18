# HoloDecoder

https://matplotlib.org/stable/tutorials/introductory/sample_plots.html

https://realpython.com/pandas-dataframe/

https://stackabuse.com/tensorflow-neural-network-tutorial/

https://www.kaggle.com/hbaderts/simple-feed-forward-neural-network-with-tensorflow

https://www.kdnuggets.com/2017/10/tensorflow-building-feed-forward-neural-networks-step-by-step.html

Handling overfitting in deep learning models:
https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e

# CUDA installation

https://docs.vmware.com/en/VMware-vSphere-Bitfusion/3.0/Example-Guide/GUID-ABB4A0B1-F26E-422E-85C5-BA9F2454363A.html

export DISTRO=ubuntu2004
export VERSION=11-0-local_11.0.3-450.51.06-1
export ARCHITECTURE=x86_64

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
sudo dpkg -i cuda-repo-ubuntu2004_11-0-local_11.0.3-450.51.06-1_<architecture>.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-<version>/7fa2af80.pub
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/<architecture>/7fa2af80.pub
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/<architecture>/7fa2af80.pub
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/<architecture>/cuda-ubuntu2004.pin
$ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-get update
sudo apt-get install cuda

Fourier transformation using ANN:
https://gist.github.com/endolith/98863221204541bf017b6cae71cb0a89

# ANN Generation 1

model1_1    5.24e-03                        0.0949
model1_2    2.75e-03                        0.1040
model1_3    3.07e-03 (high noise)           0.1005
model2      1.75e-03                        0.0934
model3      2.49e-01 (almost no learning)   0.2479
model4_1    2.49e-01 (almost no learning)   0.2487
model4_2    2.49e-01 (almost no learning)   0.2487

train_model("set_256_10000_1.json", "model1_1", 0.5, 0.001)
train_model("set_256_10000_1.json", "model1_2", 0.5, 0.003)
train_model("set_256_10000_1.json", "model1_3", 0.5, 0.006)
train_model("set_256_10000_1.json", "model2", 1, 0.003) 
train_model("set_256_10000_1.json", "model3", 2, 0.003) 
train_model("set_256_10000_1.json", "model4_1", 5, 0.001)
train_model("set_256_10000_1.json", "model4_2", 5, 0.003)

Conclusions:
1) 250 epoch is enough for fast evaluation
2) Even 128-50-25-5 is too large (or the set is too small)

# ANN Generation 2

Set size        20000
Range of errors 0:1
Batch size      64

Lowest error    g2_model3_3
architecture    200-60-20 
rate            0.006
loss            0.004
val_loss        0.069 ( 17.7 )

Smallest NN     g2_model1_1
architecture    50-15-5 
rate            0.001
loss            0.036
val_loss        0.095 ( 24.3 )

Conclusions:
1) More than 2000 epoch is not viable for training since after tahat threshold NN is no longer training. The higher size of NN the higher the effect
2) Better training set is required

# ANN Generation 3 (many programm erros were fixed here)

Set size        60000
Range of errors 0:1
Batch size      64
Set             decoded

Lowest error    g3_model3_3
architecture    200-60-20
rate            0.006
loss            0.013 
val_loss        0.118 ( 30.7 )

Observations:
1) All NN show increase in val_loss after 50 iteration. In small NN with high learning rate 0.006 the loss begins to drop after 200 epochs
2) Huge overfitting from the very beginning

Conclusions:
1) Early slopping is not viable option due to (2)

Ideas:
1) Increase set size
2) Try Droppout/Regularization
3) Make model simplier
4) Remove errors in range [0.45;0.55] (those are almoust 100% noise)
5) Try training on discrete input

# ANN Generation 3'

Set size        60000
Range of errors 0:1
Batch size      64
Set             discrete

Lowest error    g3'_model3_2
architecture    200-60-20
rate            0.003
loss            0.009 
val_loss        0.122 ( 31.2 )

Smallest NN     g3'_model1_2
architecture    50-15-5 
rate            0.003
loss            0.057
val_loss        0.142 ( 36.4 )

Observations:
1) First time when model with learning 0.003 shown the best val_loss
2) val_loss generally higher compared to decoded set

Conclusions:
1) "Try training on discrete input" has failed

Ideas:
1) Try different activation function which would be better fit for discrete (binary) input data

# ANN Generation 3 (next)

Set             v2_0_1_set_256_60000_[0.4:0.6]_7-12.pickle
Range of errors 0:0.4 AND 0.6:1.0
Batch size      64
Set             decoded
Droppout        0.2 (from second layer)
Loss            mean_absolute_error AND mean_absolute_error

Models:         
g3_e2000_lmean_absolute_error_d0.2_model1_1
g3_e2000_lmean_absolute_error_d0.2_model1_2
g3_e2000_lmean_absolute_error_d0.2_model1_3
g3_e2000_lmean_absolute_error_d0.2_model2_1
g3_e2000_lmean_absolute_error_d0.2_model2_2
g3_e2000_lmean_absolute_error_d0.2_model2_3
g3_e2000_lmean_absolute_error_d0.2_model3_1
g3_e2000_lmean_absolute_error_d0.2_model3_2
g3_e2000_lmean_absolute_error_d0.2_model3_3
g3_e2000_lmean_squared_error_d0.2_model1_1
g3_e2000_lmean_squared_error_d0.2_model1_2
g3_e2000_lmean_squared_error_d0.2_model1_3
g3_e2000_lmean_squared_error_d0.2_model2_1
g3_e2000_lmean_squared_error_d0.2_model2_2
g3_e2000_lmean_squared_error_d0.2_model2_3
g3_e2000_lmean_squared_error_d0.2_model3_1
g3_e2000_lmean_squared_error_d0.2_model3_2
g3_e2000_lmean_squared_error_d0.2_model3_3

mean_absolute_error: 

Lowest error    g3_e2000_lmean_absolute_error_d0.2_model3_1
architecture    200-60-20
rate            0.003
loss            0.006
val_loss        0.059 ( 15.1 )

Smallest NN     g3_e2000_lmean_absolute_error_d0.2_model1_2
architecture    50-15-5
rate            0.006
loss            0.039
val_loss        0.085 ( 21.8 )

mean_squared_error:

Lowest error    g3_e2000_lmean_squared_error_d0.2_model3_3
architecture    200-60-20
rate            0.012
loss            0.000
val_loss        0.019 ( 35.28 )

Smallest NN     g3_e2000_lmean_squared_error_d0.2_model1_1
architecture    50-15-5
rate            0.012
loss            0.003 ( 14.0 )
val_loss        0.023 ( 38.8 )

Observations:
1) Using MSE creates huge error compared to MAE

Ideas:
1) Use custom loss function that will give huge error if MAE crosses 0.5/256
2) Use bigger training set

# processes = []
# set = load("v2_0_1_set_256_60000_[0.4:0.6]_7-12.pickle")
# loss_func = "mean_squared_error"
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

# ANN Generation 4