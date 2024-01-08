import numpy as np
import tensorflow as tf
import datetime
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.python.keras import layers, models
from tensorflow import keras
import matplotlib.pyplot as plt


save_path = "C:/Users/13764/Documents/academics/zhejiang/GP-WP/GP-WP data/dataset_14_Logp.npy"
temp = np.load(save_path, allow_pickle = True)
res_dict_dataset = temp.item()
data_output = res_dict_dataset["data_output"]
train_output = res_dict_dataset["train_output"]
test_output = res_dict_dataset["test_output"]
data_input = res_dict_dataset["data_input"]
train_input = res_dict_dataset["train_input"]
test_input = res_dict_dataset["test_input"]

# adds a dummy dimension for cnn
train_input = train_input[..., None]
test_input = test_input[...,None]
data_input = data_input[..., None]
print(train_input.shape)

# parameters
filters = 32
kernel_size = 3
dense_nodes = 512
epochs = 300

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# cnn model
model = models.Sequential()
model.add(layers.Conv1D(filters, kernel_size, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(dense_nodes, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss=tf.keras.metrics.mean_squared_error, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
model.fit(train_input, train_output, epochs=epochs, validation_data = (test_input, test_output), callbacks=[tensorboard_callback])

# calculate errors
predicted_test_output = model.predict(test_input).flatten()
rms = mean_squared_error(test_output, predicted_test_output, squared=False)
r2 = r2_score(test_output, predicted_test_output)
predicted_data_output = model.predict(data_input).flatten()
total_rms = mean_squared_error(data_output, predicted_data_output, squared=False)
total_r2 = r2_score(data_output, predicted_data_output)
p1 = 0
p5 = 0
p10 = 0
for i in range(predicted_data_output.size):
    if data_output[i] != 0:
        error = (abs(data_output[i] - predicted_data_output[i]) / predicted_data_output[i])
        if error < 0.01:
            p1 += 1
        if error < 0.05:
            p5 += 1
        if error < 0.1:
            p10 += 1
p1 /= data_output.size
p5 /= data_output.size
p10 /= data_output.size
print(rms, r2, total_rms, total_r2, p1, p5, p10)
