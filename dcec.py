
from time import time
import numpy as np

from sklearn.cluster import KMeans

import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Input, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model

import metrics

def CAE(input_shape=(28, 28, 1), filters=[32, 64, 128, 10]):
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


class DCEC(object):
    def __init__(self, input_shape, filters=[32, 64, 128, 10], n_clusters=10, alpha=1.0):
        super(DCEC, self).__init__()
        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = True
        self.y_pred = []

        self.cae = CAE(input_shape, filters)
        hidden = self.cae.get_layer(name='embedding').output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DCEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(hidden)
        self.model = Model(inputs=self.cae.input, outputs=[clustering_layer, self.cae.output])

    def pretrain(self, x, batch_size=256, epochs=200, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining...')
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            # featurewise_center=True,
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
        )

        self.cae.compile(optimizer=optimizer, loss='mse')
        # begin training
        t0 = time()
        # self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=2)
        self.cae.model.fit_generator(datagen.flow(x, x, batch_size=batch_size), steps_per_epoch=len(x) / batch_size + 1, epochs=epochs, verbose=2)
        print('Pretraining time: ', time() - t0)
        self.cae.save(save_dir + '/pretrain_cae_model.h5')
        print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def extract_feature(self, x):  # extract features from before clustering layer
        return self.encoder.predict(x)

    def predict(self, x):
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, loss=['kld', 'mse'], loss_weights=[1, 1], optimizer='adam'):
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(self, x, y=None, batch_size=256, maxiter=2e4, tol=1e-3, update_interval=140, cae_weights=None, save_dir='./results/temp'):
        save_interval = x.shape[0] / batch_size * 5
        print('update_interval:{}， maxiter:{}， Save interval:{}'.format(update_interval, maxiter, save_interval))

        # Step 1: pretrain if necessary
        t0 = time()
        if not self.pretrained and cae_weights is None:
            print('...pretraining CAE using default hyper-parameters:')
            print('   optimizer=\'adam\';   epochs=200')
            self.pretrain(x, batch_size, save_dir=save_dir)
            self.pretrained = True
        elif cae_weights is not None:
            self.cae.load_weights(cae_weights)
            print('cae_weights is loaded successfully.')

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        self.y_pred = kmeans.fit_predict(self.encoder.predict(x))
        # 用自编码器测试指标
        result = metrics.eval(y, self.y_pred)  # 输出指标 acc, nmi, ari, f1
        print('acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f, precision: %.4f, recall: %.4f' % (result['acc'], result['nmi'], result['ari'], result['f1'], result['precision'], result['recall']))

        y_pred_last = np.copy(self.y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        # logging file
        import csv, os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logfile = open(save_dir + '/dcec_log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'f1', 'precision', 'recall', 'L', 'Lc', 'Lr'])
        logwriter.writeheader()

        t2 = time()
        loss = [0, 0, 0]
        index = 0
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)
                if y is not None:
                    result = metrics.eval(y, self.y_pred)  # 输出指标 acc, nmi, ari, f1
                    acc, nmi, ari, f1, precision, recall = result['acc'], result['nmi'], result['ari'], result['f1'], result['precision'], result['recall']
                    loss = np.round(loss, 5)
                    logdict = dict(iter=ite, acc=acc, nmi=nmi, ari=ari, f1=f1, precision=precision, recall=recall, L=loss[0], Lc=loss[1], Lr=loss[2])
                    logwriter.writerow(logdict)
                    print('Iter', ite, ': Acc', acc, ', nmi', nmi, ', ari', ari, ', f1', f1, ', precision', precision, ', recall', recall, '; loss=', loss)

                # check stop criterion
                delta_label = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    logfile.close()
                    break

            # train on batch
            if (index + 1) * batch_size > x.shape[0]:
                loss = self.model.train_on_batch(x=x[index * batch_size::],
                                                 y=[p[index * batch_size::], x[index * batch_size::]])
                index = 0
            else:
                loss = self.model.train_on_batch(x=x[index * batch_size:(index + 1) * batch_size],
                                                 y=[p[index * batch_size:(index + 1) * batch_size],
                                                    x[index * batch_size:(index + 1) * batch_size]])
                index += 1
            ite += 1

        # save the trained model
        logfile.close()
        print('saving model to:', save_dir + '/dcec_model_final.h5')
        self.model.save(save_dir + '/dcec_model_final.h5')
        t3 = time()
        print('Pretrain time:  ', t1 - t0)
        print('Clustering time:', t3 - t1)
        print('Total time:     ', t3 - t0)


if __name__ == "__main__":
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--dataset', default='usps', choices=['usps'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=4e3, type=int)
    parser.add_argument('--gamma', default=0.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=140, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--cae_weights', default=None, help='This argument must be given')
    parser.add_argument('--save_dir', default='results/DCEC')
    args = parser.parse_args()
    args.save_dir = 'results/DCEC/{}'.format(args.dataset)
    args.cae_weights = 'results/CAE/{}/{}-pretrain-model-400.h5'.format(args.dataset, args.dataset)
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

    # prepare the DCEC model
    dcec = DCEC(input_shape=x.shape[1:], filters=[32, 64, 128, n_clusters], n_clusters=n_clusters)
    plot_model(dcec.model, to_file=args.save_dir + '/dcec_model.png', show_shapes=True)
    dcec.model.summary()

    # begin clustering.
    optimizer = 'adam'
    dcec.compile(loss=['kld', 'mse'], loss_weights=[args.gamma, 1], optimizer=optimizer)
    dcec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter, update_interval=args.update_interval, save_dir=args.save_dir, cae_weights=args.cae_weights)
    y_pred = dcec.y_pred
    result = metrics.eval(y, y_pred)  # 输出指标 acc, nmi, ari, f1
    print('acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f, precision: %.4f, recall: %.4f' % (
    result['acc'], result['nmi'], result['ari'], result['f1'], result['precision'], result['recall']))


