import iree.compiler.tf
import iree.runtime
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
BATCH_SIZE = 32

def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape into grayscale images:
    x_train = np.reshape(x_train, (-1, NUM_ROWS, NUM_COLS, 1))
    x_test = np.reshape(x_test, (-1, NUM_ROWS, NUM_COLS, 1))

    # Rescale uint8 pixel values into float32 values between 0 and 1:
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    # IREE doesn't currently support int8 tensors, so we cast them to int32:
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return (x_train, y_train), (x_test, y_test)