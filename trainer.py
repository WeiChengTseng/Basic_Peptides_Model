import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

class Trainer(object):
    def __init__(self, sess, num_training_data=20000, epoch=None, batch_size=None, print_every=100):
        '''
        Initialize the trainer.

        Input:
        - sess: tf.Session()
        - num_training_data: the number of training data.
        - epoch: the number of training epoch.
        - batch_size: batch size.
        - print_every: how often we print the result.
        '''

        self.epoch = epoch
        self.num_training_data = num_training_data
        self.batch_size = batch_size
        self.num_batch = int(num_training_data / batch_size)
        self.print_every = print_every
        self.sess = sess

        return

    def train(self, training_data, testing_data, objective_fun, prediction, threshold, x, y, keep_prob, batch_size):
        """
        Train the model.
        
        Input:
        - training_data: a function which returns training data.
        - testing_data: a function which returns testing data.
        - objective_fun: the tensor representing the objective function.
        - prediction: a tensor which contains the prediction of the model.
        - threshold: the threshold for precision and recall.
        - x: tf.placeholder that contains peptides sequence.
        - y: tf.placeholder that contains ground truth label.
        - keep_prob: the keep probability of dropout.
        - batch_size: batch size.
        """

        learning_rate = 0.0001
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(objective_fun)

        pre = self.precision([0.4], y, prediction)
        re = self.recall([0.4], y, prediction)

        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        x_test, y_test = testing_data()
        
        writer_train = tf.summary.FileWriter('logs/train/', self.sess.graph)
        writer_test = tf.summary.FileWriter('logs/test/')
        for epoch in range(self.epoch):
            for batch in range(self.num_batch):
                x_batch, y_batch = training_data(batch, batch_size)   
                summary_train,_ = self.sess.run([merged, train_step], feed_dict={x: x_batch, y: y_batch, keep_prob: 0.5})
                if (batch+1) % 10 == 0:
                    print('batch: '+str(batch)+', loss:', self.sess.run(objective_fun, feed_dict = {x: x_batch, y: y_batch, keep_prob: 0.5}))
                    if (batch+1) % self.print_every == 0:
                        precision, recall, index = 0, 0, ((epoch*self.num_batch + batch + 1) // self.print_every)
                        for i in range(125):
                            precision += self.sess.run(pre, feed_dict={x: x_test[64*i: 64*i+64], y: y_test[64*i: 64*i+64],  keep_prob: 1})
                            recall += self.sess.run(re, feed_dict={x: x_test[64*i: 64*i+64], y: y_test[64*i: 64*i+64],  keep_prob: 1})
                        print('\033[93m' + 'epoch: ' + str(epoch) + '\033[0m')
                        print('\033[93m' + 'precesion: ' + str(precision/125) + ', recall: ' + str(recall/125) + '\033[0m')

                        writer_train.add_summary(summary_train, ((epoch*self.num_batch + batch + 1) // 20))
                        summary_test = self.sess.run(merged, feed_dict={x: x_test[:64], y: y_test[:64], keep_prob: 1})
                        writer_test.add_summary(summary_test, ((epoch*self.num_batch + batch + 1) // 20))
    
        saver.save(self.sess,"nets/my_net.ckpt")
        return

    def precision(self, thresholds, labels, prediction, num_partition=8):
        """
        Calculate the precision of the prediction.

        Input:
        - thresholds: a list of thresholds that are applied to the precision.
        - labels: ground truth of the prediction.
        - prediction: the prediction of the model.
        - num_partition: the number of part we need to seperate to compute precision.

        Output:
        - precision: a tensor that represent the precision.
        """
        
        partition = int(self.batch_size / num_partition)
        with tf.name_scope('precision'):
            _, pre_0 = tf.metrics.precision_at_thresholds(labels[: partition], prediction[: partition], thresholds)
            _, pre_1 = tf.metrics.precision_at_thresholds(labels[partition: 2*partition], prediction[partition: 2*partition], thresholds)
            _, pre_2 = tf.metrics.precision_at_thresholds(labels[2*partition: 3*partition], prediction[2*partition: 3*partition], thresholds)
            _, pre_3 = tf.metrics.precision_at_thresholds(labels[3*partition: 4*partition], prediction[3*partition: 4*partition], thresholds)
            _, pre_4 = tf.metrics.precision_at_thresholds(labels[4*partition: 5*partition], prediction[4*partition: 5*partition], thresholds)
            _, pre_5 = tf.metrics.precision_at_thresholds(labels[5*partition: 6*partition], prediction[5*partition: 6*partition], thresholds)
            _, pre_6 = tf.metrics.precision_at_thresholds(labels[6*partition: 7*partition], prediction[6*partition: 7*partition], thresholds)
            _, pre_7 = tf.metrics.precision_at_thresholds(labels[7*partition: self.batch_size], prediction[7*partition: self.batch_size], thresholds)

            precision = tf.reduce_mean(tf.stack([pre_0, pre_1, pre_2, pre_3, pre_4, pre_5, pre_6, pre_7]))
        tf.summary.scalar('precesion', precision)

        return precision
        
    def recall(self, thresholds, labels, prediction, num_partition=8):
        """
        Calculate the recall of the prediction.

        Input:
        - thresholds: a list of thresholds that are applied to the recall.
        - labels: ground truth of the prediction.
        - prediction: the prediction of the model.
        - num_partition: the number of part we need to seperate to compute precision.

        Output:
        - recall: a tensor that represent the recall.
        """

        partition = int(self.batch_size / num_partition)
        with tf.name_scope('recall'):
            _, re_0 = tf.metrics.recall_at_thresholds(labels[: partition], prediction[: partition], thresholds)
            _, re_1 = tf.metrics.recall_at_thresholds(labels[partition: 2*partition], prediction[partition: 2*partition], thresholds)
            _, re_2 = tf.metrics.recall_at_thresholds(labels[2*partition: 3*partition], prediction[2*partition: 3*partition], thresholds)
            _, re_3 = tf.metrics.recall_at_thresholds(labels[3*partition: 4*partition], prediction[3*partition: 4*partition], thresholds)
            _, re_4 = tf.metrics.recall_at_thresholds(labels[4*partition: 5*partition], prediction[4*partition: 5*partition], thresholds)
            _, re_5 = tf.metrics.recall_at_thresholds(labels[5*partition: 6*partition], prediction[5*partition: 6*partition], thresholds)
            _, re_6 = tf.metrics.recall_at_thresholds(labels[6*partition: 7*partition], prediction[6*partition: 7*partition], thresholds)
            _, re_7 = tf.metrics.recall_at_thresholds(labels[7*partition: self.batch_size], prediction[7*partition: self.batch_size], thresholds)

        recall = tf.reduce_mean(tf.stack([re_0, re_1, re_2, re_3, re_4, re_5, re_6, re_7]))
        tf.summary.scalar('recall', recall)

        return recall


    def AUC(self, labels, prediction):
        """
        Calculate the AUC of the prediction.

        Input:
        - labels: ground truth of the prediction
        - prediction: the prediction of the model

        Output:
        - AUC: a tensor that represent the AUC.
        """
        _, auc = tf.metrics.auc(labels, prediction)
        tf.summary.scalar('AUC', auc)

        return auc

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
