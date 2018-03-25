import tensorflow as tf
import numpy as np
import math
import os

class Model():
    def __init__(self, num_label, word_dim=10, batch_size=32):
        self.num_label = num_label
        self.word_dim = word_dim
        self.batch_size = batch_size

        return

    def build(self, x, y, reg, keep_prob):
        """
        Build the model.

        Input:
        - x: the input data, that is, the peptide sequences.
        - y: the ground truth of the peptides.
        - reg: the weight of the regression.
        Output:
        - loss: the loss of the model.
        - logits: the result of the logit regression.
        - predict: the prediction of the peptides.
        """
        logits, params = self.sub_model(x, keep_prob)
        loss = self.loss(y, logits, params, reg)
        predict = self.predict(logits)

        return loss, logits, predict

    def sub_model(self, x, keep_prob):
        """
        Define the architecture of the model.

        Input:
        - x: the input data, that is, the peptide sequences.
        - keep_prob: the keep probability of dropout.

        Output:
        - logits: the result of the logit regression.
        - params: some weights and filters used in the model.
        """

        params = []
        with tf.name_scope('filters'):
            Filter1 = tf.Variable(tf.truncated_normal([6, self.word_dim, 128], stddev = 0.1), name = 'Filter_1')
            Filter2 = tf.Variable(tf.truncated_normal([6, 128, 128], stddev = 0.1), name = 'Filter_2')
            Filter3 = tf.Variable(tf.truncated_normal([5, 128, 256], stddev = 0.1), name = 'Filter_3')
            Filter4 = tf.Variable(tf.truncated_normal([5, 256, 256], stddev = 0.1), name = 'Filter_4')
            Filter5 = tf.Variable(tf.truncated_normal([5, 256, 512], stddev = 0.1), name = 'Filter_5')
            Filter6 = tf.Variable(tf.truncated_normal([5, 512, 512], stddev = 0.1), name = 'Filter_6')
            self.variable_summaries(Filter1)
            self.variable_summaries(Filter2)
            self.variable_summaries(Filter3)
            self.variable_summaries(Filter4)
            self.variable_summaries(Filter5)
            self.variable_summaries(Filter6)

        with tf.name_scope('weights'):
            W7 = tf.Variable(tf.truncated_normal([7168, 1024], stddev = 0.1), name = 'W7')
            W8 = tf.Variable(tf.truncated_normal([1024, self.num_label], stddev = 0.1), name = 'W8')
            self.variable_summaries(W7)
            self.variable_summaries(W8)

        with tf.name_scope('bias'):
            b1 = tf.Variable(tf.zeros([128]), name = 'b1')
            b2 = tf.Variable(tf.zeros([128]), name = 'b2')
            b3 = tf.Variable(tf.zeros([256]), name = 'b3')
            b4 = tf.Variable(tf.zeros([256]), name = 'b4')
            b5 = tf.Variable(tf.zeros([512]), name = 'b5')
            b6 = tf.Variable(tf.zeros([512]), name = 'b6')
            b7 = tf.Variable(tf.zeros([1024]), name = 'b7')
            b8 = tf.Variable(tf.zeros([self.num_label]), name = 'b8')
            self.variable_summaries(b1)
            self.variable_summaries(b2)
            self.variable_summaries(b3)
            self.variable_summaries(b4)
            self.variable_summaries(b5)
            self.variable_summaries(b6)
            self.variable_summaries(b7)
            self.variable_summaries(b8)
        alpha = 0.2

        with tf.name_scope('Conv_1'):
            L1 = tf.nn.conv1d(x, Filter1, stride = 1, padding = 'VALID', data_format='NHWC') + b1
        with tf.name_scope('leaky_relu_1'):
            L1_act = tf.nn.leaky_relu(L1, alpha)
        L1_bn = tf.layers.batch_normalization(L1_act, scale = False, name = 'bn_1')

        with tf.name_scope('Conv_2'):
            L2 = tf.nn.conv1d(L1_bn, Filter2, stride = 1, padding = 'VALID') + b2
        with tf.name_scope('leaky_relu_2'):
            L2_act = tf.nn.leaky_relu(L2, alpha)
        L2_pooled = tf.layers.max_pooling1d(L2_act, pool_size = 2, strides = 2, name = 'max_pool_2')
        L2_bn = tf.layers.batch_normalization(L2_pooled, scale = False, name = 'bn_2')

        with tf.name_scope('Conv_3'):    
            L3 = tf.nn.conv1d(L2_bn, Filter3, stride = 1, padding = 'VALID') + b3
        with tf.name_scope('leaky_relu_3'):
            L3_act = tf.nn.leaky_relu(L3, alpha)
        L3_pooled = tf.layers.max_pooling1d(L3_act, pool_size = 2, strides = 2, name = 'max_pool_3')
        L3_bn = tf.layers.batch_normalization(L3_pooled, scale = False, name = 'bn_3')

        with tf.name_scope('Conv_4'):  
            L4 = tf.nn.conv1d(L3_bn, Filter4, stride = 1, padding = 'VALID') + b4
        with tf.name_scope('leaky_relu_4'):
            L4_act = tf.nn.leaky_relu(L4, alpha)
        L4_pooled = tf.layers.max_pooling1d(L4_act, pool_size = 2, strides = 2, name = 'max_pool_4')
        L4_bn = tf.layers.batch_normalization(L4_pooled, scale = False, name = 'bn_4')

        with tf.name_scope('Conv_5'):  
            L5 = tf.nn.conv1d(L4_bn, Filter5, stride = 1, padding = 'VALID') + b5
        with tf.name_scope('leaky_relu_5'):
            L5_act = tf.nn.leaky_relu(L5, alpha)
        L5_pooled = tf.layers.max_pooling1d(L5_act, pool_size = 2, strides = 2, name = 'max_pool_5')
        L5_bn = tf.layers.batch_normalization(L5_pooled, scale = False, name = 'bn_5')

        with tf.name_scope('Conv_6'):  
            L6 = tf.nn.conv1d(L5_bn, Filter6, stride = 1, padding = 'VALID') + b6
        with tf.name_scope('leaky_relu_6'):
            L6_act = tf.nn.leaky_relu(L6, alpha)
        L6_pooled = tf.layers.max_pooling1d(L6_act, pool_size = 2, strides = 2, name = 'max_pool_6')
        L6_bn = tf.layers.batch_normalization(L6_pooled, scale = False, name = 'bn_6')
 
        reshaped_data = tf.reshape(L6_bn, shape = (self.batch_size, -1), name = 'reshape')

        with tf.name_scope('full_connected_7'):
            L7 = tf.matmul(reshaped_data, W7) + b7
        with tf.name_scope('leaky_relu_7'):
            L7_act = tf.nn.leaky_relu(L7, alpha)

        L7_dropout = tf.nn.dropout(L7_act, keep_prob=keep_prob, name = 'dropout')
        L7_bn = tf.layers.batch_normalization(L7_dropout, scale = True, name = 'bm_7')
            
        with tf.name_scope('full_connected_8'):
            L8 = tf.matmul(L7_bn, W8) + b8

        logits = L8
        params += [Filter1, Filter2, Filter3, Filter4, Filter5, Filter6]
        params += [W7, W8]
        return logits, params

    def predict(self, logits):
        """
        Predict the labels according to the model.

        Input:
        - logits: the result of the logit regression.

        Output:
        - x: the result of the prediction
        """
        x = tf.nn.sigmoid(logits)
        
        return x

    def loss(self, labels, logits, params, reg):
        """
        Define the loss of the model.

        Input:
        - label: the ground truth of the prediction.
        - logits: the result of the logit regression.
        - params: some weights and filters used in the model.
        - reg: the weight of the L2 loss

        Output:
        - loss: the loss of the model.
        """

        L2_loss_list = list(map(tf.nn.l2_loss, params))
        L2_loss = tf.add_n(L2_loss_list)
        loss = tf.losses.sigmoid_cross_entropy(labels, logits) + L2_loss * reg
        tf.summary.scalar('loss', loss)
        return loss
    
    def variable_summaries(self, var):
        """
        Define the tensorboard scalar and histogram summary.

        Input:
        - var: the variable we want to summarize in tensorboard.
        """
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
                tf.summary.scalar('stddev',stddev)
                tf.summary.scalar('max',tf.reduce_max(var))
                tf.summary.scalar('min',tf.reduce_min(var))
                tf.summary.histogram('histogram',var)
        return

