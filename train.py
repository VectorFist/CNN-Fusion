from fusion_model import FusionModel
from cifar_data_hls import read_data_sets as cifar_hls_read_data

import tensorflow as tf


def train_model():
    train_params = dict()
    train_params['n_epochs'] = 12
    train_params['initial_learning_rate'] = 0.001
    train_params['keep_prob'] = 0.8
    train_params['reduce_lr_epoch_1'] = 7
    train_params['reduce_lr_epoch_2'] = 10
    train_params['batch_size'] = 64

    print('prepare data...')
    cifar_data_hls_provider = cifar_hls_read_data(read=True)
    print('build graph...')
    h_patch_size = 32
    w_patch_size = 32
    model = FusionModel(cifar_data_hls_provider, [h_patch_size, w_patch_size, 4], [h_patch_size, w_patch_size, 3])
    print('train model...')
    model.train_all_epoch(train_params)
    model.save_model()


if __name__ == '__main__':
    print(tf.__version__)
    train_model()
