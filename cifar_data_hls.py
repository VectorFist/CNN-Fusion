import pickle
import numpy as np
from collections import namedtuple
import random
import time
import cv2
from multiprocessing import Pool
from matplotlib import pyplot as plt
from scipy import misc
Datasets = namedtuple('Datasets', ['train', 'test', 'name'])


def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


class DataSet(object):
    def __init__(self, images, reshape=True):
        if reshape:
            images = np.reshape(images, (images.shape[0], 3, 32, 32))
            images = images.transpose((0, 2, 3, 1))

        self._num_examples = images.shape[0]
        self._images = images
        self._gray_images = self.create_gray_images(self._images)
        self._low_res_images = self.create_low_resolution_images(self._images)
        self._images = self._images.astype(np.float32) / 255.0
        self._data = np.concatenate((self._low_res_images, self._gray_images[..., np.newaxis]), axis=3)
        self._target = self._images #- self._gray_images[..., np.newaxis] - self._low_res_images
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def gray_images(self):
        return self._gray_images

    @property
    def low_res_images(self):
        return self._low_res_images

    @property
    def data(self):
        return self._data

    @property
    def target(self):
        return self._target

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def create_gray_images(self, _images):
        gray_images = []
        for i in range(self.num_examples):
            gray = cv2.cvtColor(_images[i], cv2.COLOR_RGB2HLS)[..., 1]
            gray_images.append(gray)
        gray_images = np.array(gray_images, dtype=np.float32) / 255.0
        return gray_images

    def create_low_resolution_images(self, _images):
        low_res_images = []
        for i in range(self.num_examples):
            half = misc.imresize(_images[i], 0.5)
            low_res_images.append(misc.imresize(half, 2.0, interp='bicubic'))
        low_res_images = np.array(low_res_images, dtype=np.float32) / 255.0
        return low_res_images

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and self._index_in_epoch == 0 and shuffle:
            self.shuffle()

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            target_rest_part = self._target[start:self._num_examples]

            if shuffle:
                self.shuffle()

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            target_new_part = self._target[start:end]
            return np.concatenate((data_rest_part, data_new_part)), np.concatenate((target_rest_part, target_new_part))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._target[start:end]

    def shuffle(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._gray_images = self._gray_images[perm]
        self._low_res_images = self._low_res_images[perm]
        self._data = self._data[perm]
        self._target = self._target[perm]
        print('shuffle done')


def read_data_sets(reshape=True, read=True):
    print('read cifar datasets...')
    if not read:
        print('read datasets done !!!')
        return Datasets(train=None, test=None, name='cifar_hls')

    data_10_dir = 'cifar/cifar-10-batches-py/'
    data_100_dir = 'cifar/cifar-100-python/'
    train_10_files = [data_10_dir+'data_batch_{:d}'.format(i) for i in range(1, 6)]
    train_100_file = data_100_dir + 'train'
    test_10_file = data_10_dir + 'test_batch'
    test_100_file = data_100_dir + 'test'

    train_10_dicts = []
    for i in range(5):
        train_10_dicts.append(unpickle(train_10_files[i]))
    test_10_dict = unpickle(test_10_file)
    train_100_dict = unpickle(train_100_file)
    test_100_dict = unpickle(test_100_file)

    train_10_data = [train_10_dicts[i][b'data'] for i in range(5)]
    train_10_data = np.concatenate(train_10_data, axis=0)
    train_100_data = train_100_dict[b'data']

    test_10_data = test_10_dict[b'data']
    test_100_data = test_100_dict[b'data']

    train_data = np.concatenate((train_10_data, train_100_data), axis=0)
    test_data = np.concatenate((test_10_data, test_100_data), axis=0)

    train = DataSet(train_data, reshape=reshape)
    test = DataSet(test_data, reshape=reshape)

    print('read datasets done !!!')
    return Datasets(train=train, test=test, name='cifar_hls')


def read_test_data_from_image_path(image_path):
    image = plt.imread(image_path)
    image = misc.imresize(image, (32, 32), interp='nearest').astype(np.float32)
    low_res_image = misc.imresize(misc.imresize(image, 0.5, interp='nearest'), 2.0).astype(np.float32)

    image = image[np.newaxis, ...] / 255.0
    low_res_image = low_res_image[np.newaxis, ...] / 255.0
    gray_image = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
    data = np.concatenate((low_res_image, gray_image[..., np.newaxis]), axis=3)
    target = image - gray_image[..., np.newaxis] - low_res_image

    return data, image, low_res_image, gray_image


if __name__ == '__main__':
    data = read_data_sets()
    print(data.train.num_examples)
    print(data.test.num_examples)
    plt.imshow(data.train.images[9])
    plt.figure()
    plt.imshow(data.train.gray_images[9], cmap='gray')
    plt.figure()
    plt.imshow(data.train.low_res_images[9])
    plt.show()
