from fusion_model import FusionModel
from cifar_data_hls import read_data_sets as cifar_hls_read_data

import tensorflow as tf


def evaluate_model():
    print('prepare data...')
    cifar_data_hls_provider = cifar_hls_read_data(read=True)
    print('build graph...')
    h_patch_size = 32
    w_patch_size = 32
    model = FusionModel(cifar_data_hls_provider, [h_patch_size, w_patch_size, 4], [h_patch_size, w_patch_size, 3])
    print('train model...')
    model.load_model()
    model.test()


if __name__ == '__main__':
    print(tf.__version__)
    evaluate_model()
