import numpy as np
import tensorflow as tf
import skimage.color
import skimage.transform
from random import sample


def preprocess(img):
    img = skimage.transform.resize(img, (30, 45))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)
   
    return tf.stack(img)


def split_tuple(samples):
    samples = np.array(samples, dtype=object)
    screen_buf = tf.stack(samples[:,0])
    actions = samples[:,1]
    rewards = tf.stack(samples[:,2])
    next_screen_buf = tf.stack(samples[:,3])
    dones = tf.stack(samples[:,4])  
    return screen_buf, actions, rewards, next_screen_buf, dones 


def extractDigits(*argv):
    if len(argv)==1:
        return list(map(lambda x: [x], argv[0]))

    return list(map(lambda x,y: [x,y], argv[0], argv[1]))


def get_samples(memory):
    if len(memory) < 64:
        sample_size = len(memory)
    else:
        sample_size = 64

    return sample(memory, sample_size)