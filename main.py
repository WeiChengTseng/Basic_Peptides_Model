import matplotlib.pyplot as plt
import tensorflow as tf
import networkx as nx
import numpy as np
import math
import time
import os

from input_ops import *
from model import *
from trainer import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
#config.log_device_placement=True

with tf.Session(config=config) as sess:
    num_training_data = 512000
    batch_size, word_dim, threshold, reg = 64, 10, 0.5, 1e-5
    num_batch = num_training_data // batch_size

    dataset = Dataset(num_training_data=num_training_data, num_testing_data=8000, word_dim=word_dim)
    x, y = tf.placeholder(tf.float32, shape=[batch_size, 608, word_dim]), tf.placeholder(tf.int8, shape=[batch_size, dataset.num_labels])
    keep_prob = tf.placeholder(tf.float32)

    model = Model(num_label=dataset.num_labels, word_dim=word_dim, batch_size=batch_size)
    trainer = Trainer(sess=sess, Model=model, num_training_data=num_training_data, epoch=30, batch_size=batch_size)

    loss, logits, predict = model.build(x, y, reg=reg, keep_prob=keep_prob)
    trainer.train(dataset.get_training_data, dataset.get_testing_data, loss, predict, threshold, x, y, keep_prob, batch_size)

    
