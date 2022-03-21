from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Input, LeakyReLU, MaxPooling2D, UpSampling2D, \
    BatchNormalization
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
from keras import backend as K
import numpy as np


# 加上 池化层 以及 LeakRelu层
def myCAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    model.add(Conv2D(filters[0], 5, strides=1, padding='same', name='conv1', input_shape=input_shape))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(filters[1], 5, strides=1, padding='same', name='conv2'))
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3'))
    model.add(Flatten())

    model.add(Dense(units=filters[3], name='embedding'))

    model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu'))
    model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))

    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, name='deconv3'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2DTranspose(filters[0], 5, strides=1, padding='same', name='deconv2'))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D((2, 2)))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=1, padding='same', name='deconv1'))  # activation='sigmoid'
    return model


if __name__ == "__main__":
    from time import time
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='usps', choices=['usps'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_dir', default='results/CAE', type=str)
    args = parser.parse_args()
    args.save_dir = 'results/CAE/{}'.format(args.dataset)
    print(args)

    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load dataset
    from datasets import load_data
    # os.chdir('../../')  # print(os.getcwd())
    x, y = load_data(args.dataset)  # 去上上级目录调数据集
    # os.chdir('./deep_cluster/MyCode')  # print(os.getcwd())

    n_clusters = len(np.unique(y))

    # 使用数据增强
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        # featurewise_center=True,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
    )

    # define the model
    model = myCAE(input_shape=x.shape[1:], filters=[32, 64, 128, n_clusters])
    model.compile(optimizer='adam', loss='mse')
    plot_model(model, to_file=args.save_dir + '/%s-pretrain-model.png' % args.dataset, show_shapes=True)
    model.summary()

    # begin training
    t0 = time()

    # 不使用数据增强
    # model.fit(x, x, batch_size=args.batch_size, epochs=args.epochs, verbose=2)
    # 使用数据增强
    model.fit_generator(datagen.flow(x, x, batch_size=args.batch_size), steps_per_epoch=len(x) / args.batch_size + 1, epochs=args.epochs, verbose=2)

    print('Training time: ', time() - t0)
    model.save(args.save_dir + '/%s-pretrain-model-%d.h5' % (args.dataset, args.epochs))

    # extract features
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name='embedding').output)
    features = feature_model.predict(x)
    print('feature shape=', features.shape)

    # use features for clustering
    features = np.reshape(features, newshape=(features.shape[0], -1))

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters)
    y_pred = km.fit_predict(features)

    import metrics
    result = metrics.eval(y, y_pred)  # 输出指标 acc, nmi, ari, f1
    print('acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f, precision: %.4f, recall: %.4f' % (result['acc'], result['nmi'], result['ari'], result['f1'], result['precision'], result['recall']))


