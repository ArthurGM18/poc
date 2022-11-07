import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense, ReLU


class CNN(Model):
    def __init__(self, num_actions):
        super(CNN,self).__init__()
        self.conv1 = Sequential([
                                Conv2D(8, kernel_size=6, strides=3, input_shape=(30,45,1)),
                                BatchNormalization(),
                                ReLU()
                                ])

        self.conv2 = Sequential([
                                Conv2D(8, kernel_size=3, strides=2, input_shape=(9, 14, 8)),
                                BatchNormalization(),
                                ReLU()
                                ])
        
        self.flatten = Flatten()
       
        self.state_value = Dense(1) 
        self.advantage = Dense(num_actions)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x1 = x[:, :96]
        x2 = x[:, 96:]
        x1 = self.state_value(x1)
        x2 = self.advantage(x2) 
        
        x = x1 + (x2 - tf.reshape(tf.math.reduce_mean(x2, axis=1), shape=(-1,1)))
        return x