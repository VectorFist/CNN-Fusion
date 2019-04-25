import os
import time
import shutil
import cv2
import math
from datetime import timedelta
from matplotlib import pyplot as plt
from scipy import misc

import numpy as np
import tensorflow as tf
import sys


class FusionModel(object):
    def __init__(self, data_provider, data_shape, target_shape):
        self.data_provider = data_provider
        self.dataset_name = data_provider.name
        self.renew_logs = True
        self.should_save_logs = True
        self.should_save_model = True
        self._define_inputs(data_shape, target_shape)
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _define_inputs(self, data_shape, target_shape):
        data_shape = [None] + data_shape
        target_shape = [None] + target_shape
        self.data_input = tf.placeholder(tf.float32, data_shape, name='data_input')
        self.target_input = tf.placeholder(tf.float32, target_shape, name='target_input')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def _build_graph(self):
        padding_h = int(self.data_input.get_shape()[1]) + 6
        padding_w = int(self.data_input.get_shape()[2]) + 6
        data_input = tf.image.pad_to_bounding_box(self.data_input, 3, 3, padding_h, padding_w)

        MS_input = data_input[..., 0:3]
        Pan_input = data_input[..., 3:4]

        MS_conv1 = self.conv2d_block(MS_input, 32, 3, 'MS_conv1')
        MS_conv2 = self.conv2d_block(MS_conv1, 64, 3, 'MS_conv2')
        MS_conv3 = self.conv2d_block(MS_conv2, 128, 3, 'MS_conv3')
        Pan_conv1 = self.conv2d_block(Pan_input, 32, 3, 'Pan_conv1')
        Pan_conv2 = self.conv2d_block(Pan_conv1, 64, 3, 'Pan_conv2')
        Pan_conv3 = self.conv2d_block(Pan_conv2, 128, 3, 'Pan_conv3')

        MS_Pan_input1 = tf.concat((MS_input, Pan_input), axis=3)
        MS_Pan_conv1 = self.conv2d_fusion_block(MS_Pan_input1, 64, 'MS_Pan_conv1')
        MS_Pan_input2 = tf.concat((MS_conv1, MS_Pan_conv1[:, 1: -1, 1:-1, :], Pan_conv1), axis=3)
        MS_Pan_conv2 = self.conv2d_fusion_block(MS_Pan_input2, 128, 'MS_Pan_conv2')
        MS_Pan_input3 = tf.concat((MS_conv2, MS_Pan_conv2[:, 1: -1, 1:-1, :], Pan_conv2), axis=3)
        MS_Pan_conv3 = self.conv2d_fusion_block(MS_Pan_input3, 256, 'MS_Pan_conv3')

        fusion_input = tf.concat((MS_conv3, MS_Pan_conv3[:, 1: -1, 1:-1, :], Pan_conv3), axis=3)
        output = self.conv2d_final_fusion(fusion_input, [256, 128, 3], 'final_fusion')
        print(output)

        self.output = output
        self.zero_frac1 = tf.nn.zero_fraction(MS_conv1)
        self.zero_frac2 = tf.nn.zero_fraction(MS_conv2)
        self.zero_frac3 = tf.nn.zero_fraction(MS_conv3)
        self.zero_frac4 = tf.nn.zero_fraction(Pan_conv1)
        self.zero_frac5 = tf.nn.zero_fraction(Pan_conv2)
        self.zero_frac6 = tf.nn.zero_fraction(Pan_conv3)
        self.zero_frac7 = tf.nn.zero_fraction(MS_Pan_conv1)
        self.zero_frac8 = tf.nn.zero_fraction(MS_Pan_conv2)
        self.zero_frac9 = tf.nn.zero_fraction(MS_Pan_conv3)

        self.loss = tf.reduce_mean(tf.square(output-self.target_input), name='l2_loss')
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def conv2d_block(self, _input, out_features, k_size, name, padding='VALID', activation_fn=tf.identity):
        with tf.variable_scope(name):
            conv1 = self.conv2d_layer(_input, out_features, k_size, 1, 'conv1', padding=padding,
                                      activation_fn=activation_fn)
            conv2 = self.conv2d_layer(conv1, out_features, 1, 1, 'conv2', padding=padding, activation_fn=activation_fn)
        return conv2

    def conv2d_fusion_block(self, _input, out_features, name, activation_fn=tf.identity):
        with tf.variable_scope(name):
            conv1 = self.conv2d_layer(_input, out_features, 1, 1, 'conv1', activation_fn=activation_fn)
            conv2 = self.conv2d_layer(conv1, out_features, 1, 1, 'conv2', activation_fn=activation_fn)
            conv3 = self.conv2d_layer(conv2, out_features, 1, 1, 'conv3', activation_fn=activation_fn)
        return conv3

    def conv2d_final_fusion(self, _input, out_features, name, activation_fn=tf.identity):
        with tf.variable_scope(name):
            conv1 = self.conv2d_layer(_input, out_features[0], 1, 1, 'conv1', activation_fn=activation_fn)
            conv2 = self.conv2d_layer(conv1, out_features[1], 1, 1, 'conv2', activation_fn=activation_fn)
            conv3 = self.conv2d_layer(conv2, out_features[2], 1, 1, 'conv3', activation_fn=tf.identity)
        self.zero_frac10 = tf.nn.zero_fraction(conv1)
        return conv3

    def conv2d_layer(self, _input, out_features, k_size, stride, name,  activation_fn, padding='VALID'):
        with tf.variable_scope(name):
            in_features = int(_input.get_shape()[-1])
            weighs = tf.get_variable(dtype=tf.float32, shape=[k_size, k_size, in_features, out_features],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1), name='weigjhts')
            biases = tf.get_variable(dtype=tf.float32, shape=[out_features],
                                     initializer=tf.constant_initializer(0.05), name='biases')
            conv = tf.nn.conv2d(_input, weighs, [1, stride, stride, 1], padding=padding, name='conv') + biases
            '''conv = tf.contrib.layers.conv2d(_input, out_features, k_size, stride=stride, padding=padding,
                                            activation_fn=activation_fn,
                                            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                            biases_initializer=tf.constant_initializer(0.05))'''
            conv = activation_fn(conv)
        return conv

    def deconv2d_layer(self, _input, out_features, k_size, stride, name, padding='valid', activation_fn=tf.identity):
        with tf.variable_scope(name):
            deconv = tf.contrib.layers.conv2d_transpose(_input, out_features, k_size,
                                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                        biases_initializer=tf.constant_initializer(0.05),
                                                        stride=stride, padding=padding, activation_fn=activation_fn)
        return deconv

    def _initialize_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.logs_path, self.sess.graph)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        self.total_parameters = total_parameters
        print('Total training params: {:.2f}M'.format(total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/{:s}'.format(self.model_identifier)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return  save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/{:s}'.format(self.model_identifier)
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return 'fusion_model'

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        self.saver.restore(self.sess, self.save_path)
        print('Seccessfully load model from save path: {:s}'.format(self.save_path))

    def log_loss(self, loss, epoch, prefix, should_print=True):
        if should_print:
            print('{:s}\tmean loss: {:.5f}'.
                  format(prefix, loss))
        summary = tf.Summary(value=[tf.Summary.Value(tag='loss_%s' % prefix, simple_value=float(loss))])
        self.summary_writer.add_summary(summary, epoch)

    def train_all_epoch(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        keep_prob = train_params['keep_prob']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            print('\n', '-'*30, 'Train epoch: {:d}'.format(epoch), '-'*30)
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print('Decrease learning rate, new lr = {:5f}'.format(learning_rate))

            loss = self.train_one_epoch(self.data_provider.train, batch_size, learning_rate, keep_prob)
            if self.should_save_logs:
                self.log_loss(loss, epoch, prefix='train')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print('Time_per_epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)), str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

        total_training_time = time.time() - total_start_time
        print('\nTotal training time: %s' % str(timedelta(seconds=total_training_time)))

    def train_one_epoch(self, train_data, batch_size, learning_rate, keep_prob):
        num_examples = train_data.num_examples
        total_loss = []
        num_batch = num_examples // batch_size
        time_start = time.time()
        for i in range(num_batch):
            batch = train_data.next_batch(batch_size)
            data, target = batch
            feed_dict = {self.data_input: data, self.target_input: target, self.learning_rate: learning_rate,
                         self.keep_prob: keep_prob}
            fetches = [self.train_step, self.loss]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss = result
            total_loss.append(loss)
            if i % 100 == 0:
                print('Train batch: {:d}/{:d}, loss: {:.8f}, time_per_batch: {:.3f}(s)'
                      .format(i, num_batch, loss, (time.time()-time_start)/100))
                time_start = time.time()
                #print(output[0][...,0])
                #print(zf1, zf2, zf3, zf4, zf5, zf6, zf7, zf8, zf9, zf10, zf11)
        mean_loss = np.mean(total_loss)
        print('mean loss: {:.8f}'.format(mean_loss))
        return mean_loss

    def test(self):
        image_index = 100
        self.show_image(self.data_provider.test, image_index)
        while True:
            next_or_last = input()
            if next_or_last == 'a':
                image_index = max(image_index - 1, 0)
            if next_or_last == 'd':
                image_index = image_index + 1
            if next_or_last == 'p':
                sys.exit(0)
            self.show_image(self.data_provider.test, image_index)

    def test_real_image(self, img_path):
        path = img_path
        img = plt.imread(path).astype(np.float32) / 255.0
        high = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HLS)[..., 1].astype(np.float32) / 255.0
        low = misc.imresize(misc.imresize(img, 0.5), high.shape).astype(np.float32) / 255.0

        data = np.concatenate((low, high[..., np.newaxis]), axis=2)[np.newaxis, ...]
        output = self.sess.run(self.output, feed_dict={self.data_input: data})[0]
        fusion = output
        fusion[fusion < 0] = 0
        fusion[fusion > 1] = 1
        print(np.sqrt(np.mean(np.square(img-fusion))))

        plt.figure('low')
        plt.imshow(low)
        plt.figure('high')
        plt.imshow(high, cmap='gray')
        plt.figure('fusion')
        plt.imshow(fusion)
        plt.figure('img')
        plt.imshow(img)
        plt.figure('diff')
        plt.imshow(img-fusion)
        plt.show()

    def test_remote_sense_image(self, pan_path, ms_path, save_path=None):
        high_path = pan_path
        low_path = ms_path
        high = plt.imread(high_path).astype(np.float32) / 255.0 / 1.1
        if len(high.shape) == 3:
            high = high[:, :, 0]
        h, w = high.shape[0], high.shape[1]

        time1 = time.time()
        low = misc.imresize(plt.imread(low_path), (h, w), interp='bicubic').astype(np.float32) / 255.0

        if len(low.shape) == 2:
            low = low.reshape((low.shape[0], low.shape[1], 1))
            low = np.concatenate((low, low, low), axis=2)

        print(high.shape, low.shape)

        data = np.concatenate((low, high[..., np.newaxis]), axis=2)[np.newaxis, ...]
        output = self.sess.run(self.output, feed_dict={self.data_input: data, self.keep_prob: 1.0})[0]
        print(time.time()-time1)

        fusion = output
        fusion[fusion <= 0] = 0
        fusion[fusion >= 1] = 1

        fu_L = cv2.cvtColor((fusion*255).astype(np.uint8), cv2.COLOR_RGB2HLS)[..., 1].astype(np.float32) / 255.0
        rmse = np.sqrt(np.mean(np.square(fu_L - high)))
        print('rmse {:.6f}'.format(rmse))
        if save_path:
            misc.imsave(save_path, fusion)

        print('use time: {:.4f}s'.format(time.time()-time1))

        plt.figure('fusion')
        plt.imshow(fusion)
        plt.figure('low')
        plt.imshow(low)
        plt.figure('high')
        plt.imshow(high, cmap='gray')
        plt.figure('diff')
        diff = np.abs(fusion - low)
        plt.imshow(diff)
        plt.show()

    def show_image(self, test_data, image_index):
        plt.close()
        data = test_data.data[image_index: image_index + 1]
        target = test_data.target[image_index: image_index + 1]
        gray_image = test_data.gray_images[image_index: image_index + 1]
        low_res_image = test_data.low_res_images[image_index: image_index + 1]
        true_image = test_data.images[image_index: image_index + 1]

        output, loss = self.sess.run([self.output, self.loss], feed_dict={self.data_input: data, self.keep_prob: 1.0,
                                                                          self.target_input: target})

        fusion_image = output
        no_fusion_image = gray_image[..., np.newaxis] + low_res_image

        fusion_image[fusion_image < 0] = 0
        fusion_image[fusion_image > 1] = 1
        no_fusion_image[no_fusion_image < 0] = 0
        no_fusion_image[no_fusion_image > 1] = 1

        plt.figure('origin')
        plt.imshow(true_image[0])
        plt.figure('fusion(mean square loss: {:.5f})'.format(loss))
        plt.imshow(fusion_image[0])
        plt.figure('gray')
        plt.imshow(gray_image[0], cmap='gray')
        plt.figure('low res')
        plt.imshow(low_res_image[0])
        plt.show()
