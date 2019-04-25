from fusion_model import FusionModel
from cifar_data_hls import read_data_sets as cifar_hls_read_data
from matplotlib import pyplot as plt

import tensorflow as tf


def fusion_image():
    pan_path = 'images/pan1.jpg'
    ms_path = 'images/ms1.jpg'
    save_path = 'images/fusion1.jpg'

    print('prepare data...')
    cifar_data_hls_provider = cifar_hls_read_data(read=False)
    print('build graph...')
    h_patch_size = plt.imread(pan_path).shape[0]
    w_patch_size = plt.imread(pan_path).shape[1]
    model = FusionModel(cifar_data_hls_provider, [h_patch_size, w_patch_size, 4], [h_patch_size, w_patch_size, 3])
    model.load_model()

    model.test_remote_sense_image(pan_path, ms_path, save_path)


if __name__ == '__main__':
    print(tf.__version__)
    fusion_image()
