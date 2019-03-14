#! /usr/bin/python
# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
# import mnist_train

import numpy as np
import struct
import matplotlib.pyplot as plt

MOVING_AVERAGE_DECAY = 0.99


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')#输入变量，把28*28的图片变成一维数组（丢失结构信息）
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')#这个是训练集的正确结果

        keep_prob = tf.placeholder("float")
        test_feed={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}             
        y = mnist_inference.inference(x, keep_prob)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state("MNIST_model/")
            saver.restore(sess, ckpt.model_checkpoint_path)
            accuracy_pre = sess.run(tf.argmax(y, 1), feed_dict=test_feed)   # estimated value
            accuracy_true = sess.run(tf.argmax(y_,1), feed_dict=test_feed)  # actual value

            print("The real value is % d ,The estimated value is % d"  % (accuracy_true[num],accuracy_pre[num]))
            return accuracy_pre[num]    # return estimated value
           

def eval_main(argv):
    global num
    num=argv
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # import MNIST dataset
    num=evaluate(mnist)
    return num

if __name__ == '__main__':
    main()
