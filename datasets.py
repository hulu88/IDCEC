import numpy as np


def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape(x.shape[0], 28, 28, 1).astype('float32')
    x = np.divide(x, 255.)
    print('MNIST samples: x.shape={}, y.shape={}, 共{}类.'.
          format(x.shape, y.shape, len(np.unique(y))))  # (70000, 28, 28, 1) (70000,)
    return x, y


def load_mnist_32():   # (28, 28, 1) -> (32, 32, 1)
    from keras.datasets import mnist
    import cv2
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x_ = []
    for i in range(x.shape[0]):
        x_.append(cv2.resize(x[i], (32, 32), interpolation=cv2.INTER_NEAREST))  # (n, 28, 28)--> (n, 32, 32)
    x = np.array(x_)
    x = x.reshape(x.shape[0], 32, 32, 1).astype('float32')
    x = np.divide(x, 255.)
    print('MNIST samples: x.shape={}, y.shape={}, 共{}类.'.
          format(x.shape, y.shape, len(np.unique(y))))  # (70000, 28, 28, 1) (70000,)
    return x, y


def load_mnist_test():
    from keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    x = x_test.reshape(-1, 28, 28, 1).astype(np.float32)
    y = y_test.astype(np.int32)
    x /= 255.
    print('MNIST_test samples: x.shape={}, y.shape={}, 共{}类.'.
          format(x.shape, y.shape, len(np.unique(y))))  # (10000, 28, 28, 1) (10000,)
    return x, y


def load_mnist_test_32():  # (28, 28, 1) -> (32, 32, 1)
    from keras.datasets import mnist
    import cv2
    (_, _), (x_test, y_test) = mnist.load_data()
    x_ = []
    for i in range(x_test.shape[0]):
        x_.append(cv2.resize(x_test[i], (32, 32), interpolation=cv2.INTER_NEAREST))  # (n, 28, 28)--> (n, 32, 32)
    x = np.array(x_)
    x = x_test.reshape(-1, 32, 32, 1).astype(np.float32)
    y = y_test.astype(np.int32)
    x /= 255.
    print('MNIST_test samples: x.shape={}, y.shape={}, 共{}类.'.
          format(x.shape, y.shape, len(np.unique(y))))  # (10000, 28, 28, 1) (10000,)
    return x, y


def load_usps(dataset='usps'):
    x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)  # (9298, 256)
    x = x.reshape(x.shape[0], 16, 16, 1)  # 卷积类型的
    y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)  # (9298,)
    print('USPS samples: x.shape={}, y.shape={}, 共{}类.'.
          format(x.shape, y.shape, len(np.unique(y))))  # (9298, 16, 16, 1) (9298,)
    return x, y


def load_YTF(dataset='YTF', resize=None):
    import h5py
    import cv2
    hf = h5py.File('data/{}.h5'.format(dataset), 'r')
    image_chw = np.asarray(hf.get('data'), dtype='float32')
    x = np.transpose(image_chw, (0, 2, 3, 1))  # chw(10000, 3, 55, 55) -> hwc(10000, 55, 55, 3) (10000,)

    if resize is None:
        pass
    else:
        print('*' * 10, 'resize')
        x_resize = []
        for i in range(x.shape[0]):
            x_resize.append(cv2.resize(x[i], (resize, resize), interpolation=cv2.INTER_NEAREST))
        x = np.array(x_resize)

    # x = x.reshape((x.shape[0], -1))
    x = x / 255.  # x = (x-np.float32(127.5)) / np.float32(127.5)
    y = np.asarray(hf.get('labels'), dtype='int32')
    print('YTF samples: x.shape={}, y.shape={}, 共{}类.'.
          format(x.shape, y.shape, len(np.unique(y))))  # (10000, 55, 55, 3) (10000,)
    return x, y


# 图像数据集的可视化, data.shape=(n_sample, w, h, channels)
def show_figure(data, label, shuffle=True, fileName=None):  # 显示前200张图片
    import matplotlib.pyplot as plt
    digit_size_h, digit_size_w, channels = data.shape[1], data.shape[2], data.shape[3]

    # 构造矩阵，用于存放图像信息
    figure = np.zeros((digit_size_h * 10, digit_size_w * 20))
    if channels == 3:  # 3通道的图像
        figure = np.zeros((digit_size_h * 10, digit_size_w * 20, channels))

    data = np.squeeze(data)  # 去掉1维, 即灰度图像中的1通道信息
    if shuffle:  # 随机打乱数据
        import random
        idx = random.sample(range(0, data.shape[0]), 200)  # 在所有样本中随机取200个样本用于显示
    else:
        idx = range(200)  # 在所有样本中取前200个样本用于显示
    print(label[idx])
    t = 0
    for i in range(10):  # 10行
        for j in range(20):  # 每行展示20个数据
            figure[i * digit_size_h: (i + 1) * digit_size_h, j * digit_size_w: (j + 1) * digit_size_w] = data[idx[t]]
            t = t + 1

    # 去除边缘空白
    plt.figure(figsize=(10, 5))
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.axis('off')
    plt.imshow(figure)

    if fileName is not None:
        plt.savefig('./data/Sample_pictures/{}.png'.format(fileName))

    plt.show()


def load_data(dataset_name):
    if dataset_name == 'mnist':
        # return load_mnist()
        return load_mnist_32()  # resize:28 -> 32
    elif dataset_name == 'mnist_test':
        return load_mnist_test()
        # return load_mnist_test_32()   # resize:28 -> 32
    elif dataset_name == 'usps':
        return load_usps()
    elif dataset_name == 'YTF':
        return load_YTF()
        # return load_YTF('YTF', 52)
    else:
        print('无效的数据集', dataset_name)
        exit(0)


if __name__ == '__main__':
    # # load dataset:
    dataset_name = 'usps'
    x, y = load_data(dataset_name)  # 获取 x, y
    print(np.min(x), np.max(x))  # 判断是否归一化数据
    print("x.shape={}, x.shape={}, 共{}类.".format(x.shape, y.shape, len(np.unique(y))))

    # 图像的可视化
    # show_figure(x, y, shuffle=False, fileName=None)  # 可视化数据集(不保存成图片, 且不打乱数据)
    # show_figure(x, y, shuffle=False, fileName=dataset_name)  # 可视化数据集

    # 查看所有数据集信息
    # for item in ['usps']:
    #     x, y = load_data(item)  # 获取 x, y
    #     # print(np.min(x), np.max(x))  # 判断是否归一化数据
    #     show_figure(x, y, shuffle=False, fileName=item)  # 可视化数据集

'''
    MNIST samples: x.shape=(70000, 28, 28, 1), y.shape=(70000,), 共10类.
    MNIST_test samples: x.shape=(10000, 28, 28, 1), y.shape=(10000,), 共10类.
    USPS samples: x.shape=(9298, 16, 16, 1), y.shape=(9298,), 共10类.
    YTF samples: x.shape=(10000, 55, 55, 3), y.shape=(10000,), 共41类.
'''
