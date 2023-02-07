# Use a trained DenseFuse Net to generate fused images

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imsave
from datetime import datetime
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
from Model import Generator
import time
import matplotlib.pyplot as plt

from skimage import transform, data
import scipy.io as scio


def generate(oe_path, ue_path, model_path, index, output_path=None, format=None):
    oe_img = imread(oe_path) / 255.0
    ue_img = imread(ue_path) / 255.0

    H, W, C = oe_img.shape
    h = H // 8 * 8
    w = W // 8 * 8
    oe_img = oe_img[0:h, 0:w, :]
    ue_img = ue_img[0:h, 0:w, :]
    oe_img = oe_img.reshape([1, h, w, C])
    ue_img = ue_img.reshape([1, h, w, C])
    shape = oe_img.shape
    print('oe img shape', oe_img.shape)

    

    with tf.Graph().as_default(), tf.Session() as sess:
        SOURCE_oe = tf.placeholder(tf.float32, shape=shape, name='SOURCE_oe')
        SOURCE_ue = tf.placeholder(tf.float32, shape=shape, name='SOURCE_ue')

        print('SOURCE_oe shape:', SOURCE_oe.shape)

        G = Generator('Generator')
        output_image = G.transform(oe_img=SOURCE_oe, ue_img=SOURCE_ue, is_training=False)

        # restore the trained model and run the style transferring
        g_list = tf.global_variables()
        saver = tf.train.Saver(var_list=g_list)

        model_save_path = model_path + 'model.ckpt'
        print(model_save_path)
        saver.restore(sess, model_save_path)

        output = sess.run(output_image, feed_dict={SOURCE_oe: oe_img, SOURCE_ue: ue_img})
        output = output[0, :, :, :]

        imsave(output_path + str(index) + format, output)

        
        
