import tensorflow as tf
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import CMAPSSDataset

flags = tf.flags
flags.DEFINE_string("weights", None, 'weights of the network')
flags.DEFINE_integer("epochs", 100, 'train epochs')
flags.DEFINE_integer("batch_size", 32, 'batch size for train/test')
flags.DEFINE_integer("sequence_length", 32, 'sequence length')
flags.DEFINE_boolean('debug', False, 'debugging mode or not')
FLAGS = flags.FLAGS

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
log_saver = logging.FileHandler("LOGS-LSTM-Keras-CMAPSS.txt")
log_saver.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
log_saver.setFormatter(log_formatter)
log_console = logging.StreamHandler()
logger.addHandler(log_saver)
logger.addHandler(log_console)

datasets = CMAPSSDataset.CMAPSSDataset(fd_number='1', batch_size=FLAGS.batch_size, sequence_length=FLAGS.sequence_length)
train_data = datasets.get_train_data()
train_feature_slice = datasets.get_feature_slice(train_data)
train_label_slice = datasets.get_label_slice(train_data)
logging.info("train_data.shape: {}".format(train_data.shape))
logging.info("train_feature_slice.shape: {}".format(train_feature_slice.shape))
logging.info("train_label_slice.shape: {}".format(train_label_slice.shape))
test_data = datasets.get_test_data()
test_feature_slice, test_label_slice = datasets.get_last_data_slice(test_data)
logging.info("test_data.shape: {}".format(test_data.shape))
logging.info("test_feature_slice.shape: {}".format(test_feature_slice.shape))
logging.info("test_label_slice.shape: {}".format(test_label_slice.shape))

timesteps = train_feature_slice.shape[1]
input_dim = train_feature_slice.shape[2]

model = tf.keras.models.Sequential()
if tf.test.is_gpu_available():
    model.add(tf.keras.layers.CuDNNLSTM(input_shape=(timesteps, input_dim), units=100, return_sequences=True, name="lstm_0"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_0"))
    model.add(tf.keras.layers.CuDNNLSTM(units=50, return_sequences=True, name="lstm_1"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_1"))
    model.add(tf.keras.layers.CuDNNLSTM(units=25, return_sequences=False, name="lstm_2"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_2"))
    model.add(tf.keras.layers.Dense(units=1, name="dense_0"))
    model.add(tf.keras.layers.Activation('linear', name="activation_0"))
else:
    model.add(tf.keras.layers.LSTM(input_shape=(timesteps, input_dim), units=100, return_sequences=True, name="lstm_0"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_0"))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, name="lstm_1"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_1"))
    model.add(tf.keras.layers.LSTM(units=25, return_sequences=False, name="lstm_2"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_2"))
    model.add(tf.keras.layers.Dense(units=1, name="dense_0"))
    model.add(tf.keras.layers.Activation('linear', name="activation_0"))

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
print(model.summary())

def plot_results(y_pred, y_true):
    num = len(y_pred)
    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.
    fig_verify = plt.figure(figsize=(60, 30))
    # plt.plot(y_pred_test, 'ro', color="red", lw=3.0)
    # plt.plot(y_true_test, 'ro', color="blue")
    X = np.arange(1, num+1)
    width = 0.35
    plt.bar(X, np.array(y_pred).reshape(num,), width, color='r')
    plt.bar(X + width, np.array(y_true).reshape(num,), width, color='b')
    plt.xticks(X)
    plt.title('Remaining Useful Life for each turbine')
    plt.ylabel('RUL')
    plt.xlabel('Turbine')
    plt.legend(['predicted', 'actual data'], loc='upper left')
    plt.show()
    fig_verify.savefig("lstm-keras-cmapss-model.png")

if __name__ == '__main__':
    if FLAGS.weights:
        if os.path.isfile(FLAGS.weights):
            model.load_weights(FLAGS.weights)
        else:
            raise ValueError("FLAGS.weights is not a valid filepath") 
    else:
        log_filepath = "tensorboard-logs"
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
        es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto')

        history = model.fit(train_feature_slice, train_label_slice, 
                    batch_size = FLAGS.batch_size, 
                    epochs = FLAGS.epochs, 
                    validation_data = (test_feature_slice, test_label_slice),
                    verbose = 2,
                    callbacks = [tb_cb, es_cb])

        weights_save_path = 'vanilla-lstm-cmapss-weights_v0.h5'
        model.save_weights(weights_save_path)
        logging.info("Model saved as {}".format(weights_save_path))


    y_pred = model.predict(test_feature_slice)
    y_true = test_label_slice
    score = model.evaluate(test_feature_slice, test_label_slice, verbose=2)

    s1 = ((y_pred - y_true)**2).sum()
    moy = y_pred.mean()
    s2 = ((y_pred - moy)**2).sum()
    s = 1 - s1/s2
    logging.info('\nEfficiency: {}%'.format(s * 100))

    plot_results(y_pred, y_true)