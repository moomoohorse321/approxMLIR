import numpy as np
import tensorflow as tf


class approxModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.hash = 0x123456
        self.anchor = tf.Variable(0.0, dtype=tf.float32)
    
    def start_knob(self):
        self.anchor.assign(self.hash)
        return self.anchor
        