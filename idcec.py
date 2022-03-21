# -*- coding:utf-8 -*-
import argparse
import sys
from time import time
import numpy as np
from keras.applications import VGG16
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.image import ImageDataGenerator

from sklearn.cluster import KMeans

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Input, LeakyReLU, Convolution2D, \
    MaxPooling2D, Dropout, MaxPool2D
from keras.models import Sequential, Model, load_model
from keras.utils.vis_utils import plot_model

import metrics


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def CNN_model(input_shape=(28, 28, 1)):
    # 定义输入
    # input_shape = (224, 224, 3)

    # 使用序贯模型(sequential)来定义
    model = Sequential(name='vgg16-sequential')

    # 第1个卷积区块(block1)
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, name='block1_conv1'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block1_pool'))

    # 第2个卷积区块(block2)
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block2_pool'))

    # 第3个区块(block3)
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2), name='block3_pool'))

    # 前馈全连接区块
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dense(1024, activation='relu', name='fc2'))
    model.add(Dense(10, activation='softmax', name='predictions'))

    # model = Sequential()
    # model.add(Convolution2D(  # 平面大小28x28，用same padding得到的和上一次一样，也是28x28，有32个特征图
    #     input_shape=input_shape,  # 只需要在第一次添加输入平面
    #     filters=32,
    #     kernel_size=5,
    #     strides=1,
    #     padding='same',
    #     activation='relu'
    # ))
    # model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    # model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
    # model.add(MaxPooling2D(2, 2, 'same'))
    # model.add(Flatten())
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))

    return model


if __name__ == '__main__':
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['usps'])
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--kmeans_LAMBDA', default=0.60, type=float)
    parser.add_argument('--save_dir', default='results/DCEC')
    args = parser.parse_args()
    args.save_dir = 'results/DCEC/{}'.format(args.dataset)
    print(args)

    if args.dataset == 'usps':
        args.kmeans_LAMBDA = 0.98

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # load dataset
    from Deep_Cluster.datasets import load_data
    # os.chdir('../../')  # print(os.getcwd())
    x, y = load_data(args.dataset)  # 去上上级目录调数据集
    # os.chdir('./deep_cluster/MyCode/')  # print(os.getcwd())
    n_clusters = len(np.unique(y))

    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
    )

    # 加载模型并聚类
    print(args.save_dir)
    model = load_model(args.save_dir+'/dcec_model_final.h5', custom_objects={"ClusteringLayer": ClusteringLayer})
    # model = load_model(args.save_dir+'/pretrain_cae_model.h5')
    # 模型可视化
    plot_model(model, to_file="temp.png", show_shapes=True)
    model.summary()

    for i in range(30):
        if i == 0:
            feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
        else:
            feature_model = net
        print(feature_model.summary())
        features = feature_model.predict(x)

        # 用提取完的特征进行聚类分析(临时查看)
        # kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
        # y_pred = kmeans.fit_predict(features)

        # use GPU to calculate the similarity matrix
        center_t = tf.placeholder(tf.float32, (None, None))
        other_t = tf.placeholder(tf.float32, (None, None))
        center_t_norm = tf.nn.l2_normalize(center_t, dim=1)
        other_t_norm = tf.nn.l2_normalize(other_t, dim=1)
        similarity = tf.matmul(center_t_norm, other_t_norm, transpose_a=False, transpose_b=True)

        # 用提取完的特征进行聚类分析
        kmeans = KMeans(n_clusters=args.n_clusters, random_state=9).fit(features)
        y_pred = kmeans.labels_  # (9298,)
        result = metrics.eval(y, y_pred)  # 输出指标 acc, nmi, ari, f1
        print("初始聚类精确度 ===========>")
        print('acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f, precision: %.4f, recall: %.4f' % (result['acc'], result['nmi'], result['ari'], result['f1'], result['precision'], result['recall']))
        distances = kmeans.transform(features)
        center_idx = np.argmin(distances, axis=0)
        centers = [features[i] for i in center_idx]

        similarities = sess.run(similarity, {center_t:centers, other_t:features})
        reliable_image_idx = np.unique(np.argwhere(similarities > args.kmeans_LAMBDA)[:, 1])
        print(str(len(reliable_image_idx))+"/"+str(features.shape[0]))
        # print(reliable_image_idx.shape, '  ====>  ', reliable_image_idx.shape/x.shape[0])

        # 查看可信赖样本的信赖度
        y_reliable = np.array([y[i] for i in reliable_image_idx])
        y_pred_reliable = np.array([y_pred[i] for i in reliable_image_idx])
        result = metrics.eval(y_reliable, y_pred_reliable)  # 输出指标 acc, nmi, ari, f1
        print("reliable ===========>")
        print('acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f, precision: %.4f, recall: %.4f' % (result['acc'], result['nmi'], result['ari'], result['f1'], result['precision'], result['recall']))
        # exit()


        sys.stdout.flush()

        images = np.array([x[i] for i in reliable_image_idx])
        labels = to_categorical([kmeans.labels_[i] for i in reliable_image_idx])

        net = CNN_model(input_shape=x.shape[1:])

        for layer in net.layers:
            layer.trainable = True

        net.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        # net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

        # net.fit(images, labels, batch_size=128, epochs=20, verbose=2)
        net.fit_generator(datagen.flow(images, labels, batch_size=args.batch_size), steps_per_epoch=len(images) / args.batch_size + 1, epochs=args.epochs, verbose=2)

        y_pred = net.predict(x)
        y_pred = [np.argmax(l) for l in y_pred]
        y_pred = np.array(y_pred)
        result = metrics.eval(y, y_pred)  # 输出指标 acc, nmi, ari, f1
        print('acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f, precision: %.4f, recall: %.4f' % (result['acc'], result['nmi'], result['ari'], result['f1'], result['precision'], result['recall']))