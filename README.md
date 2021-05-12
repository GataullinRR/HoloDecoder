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