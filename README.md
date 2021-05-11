# HoloDecoder

https://matplotlib.org/stable/tutorials/introductory/sample_plots.html

https://realpython.com/pandas-dataframe/

https://stackabuse.com/tensorflow-neural-network-tutorial/

https://www.kaggle.com/hbaderts/simple-feed-forward-neural-network-with-tensorflow

https://www.kdnuggets.com/2017/10/tensorflow-building-feed-forward-neural-networks-step-by-step.html

# ANN Generation 1

Winner:
256 - 100 - 50 - 10 - 1 / Adam / 0.003 / mean_absolute_error

model1_1    5.24e-03
model1_2    2.75e-03
model1_3    3.07e-03 (high noise)
model2      1.75e-03
model3      2.49e-01 (almost no learning)
model4_1    2.49e-01 (almost no learning)
model4_2    2.49e-01 (almost no learning)

train_model("set_256_10000_1.json", "model1_1", 0.5, 0.001)
train_model("set_256_10000_1.json", "model1_2", 0.5, 0.003)
train_model("set_256_10000_1.json", "model1_3", 0.5, 0.006)
train_model("set_256_10000_1.json", "model2", 1, 0.003) 
train_model("set_256_10000_1.json", "model3", 2, 0.003) 
train_model("set_256_10000_1.json", "model4_1", 5, 0.001)
train_model("set_256_10000_1.json", "model4_2", 5, 0.003)