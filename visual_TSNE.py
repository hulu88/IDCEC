# -*- coding:utf-8 -*-

import time
import numpy as np
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA   # 用于降维
from sklearn.manifold import TSNE  # 用于降维

import metrics

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    dataset = 'usps'

    # from datasets import load_data  # 加载数据集
    from Deep_Cluster.datasets import load_data  # 卷积操作时使用
    x, y = load_data(dataset)
    n_cluster = len(np.unique(y))

    # 截取模型
    x = x.reshape(x.shape[0], -1)  # 转成1维
    x_2dim = TSNE(n_components=2).fit_transform(x)  # 用 TSNE 降成 2 维
    x_2dim_PCA = PCA(n_components=2).fit_transform(x.reshape(x.shape[0], -1))  # 用 PCA 降维

    # 用提取完的特征进行聚类分析
    kmeans = KMeans(n_clusters=n_cluster, n_init=20)
    y_pred = kmeans.fit_predict(x)
    result = metrics.eval(y, y_pred)  # 输出指标 acc, nmi, ari, f1
    print("初始聚类精确度 ===========>")
    print('acc: %.4f, nmi: %.4f, ari: %.4f, f1: %.4f' % (result['acc'], result['nmi'], result['ari'], result['f1']))

    # 可视化
    fig = plt.figure(figsize=(26, 32))
    ax1 = fig.add_subplot(121)
    ax1.scatter(x_2dim[:, 0], x_2dim[:, 1], c=y, s=n_cluster, label=y)
    ax1.set_xticks([])  # 去掉x轴
    ax1.set_yticks([])  # 去掉y轴
    ax1.set_title("TSNE")

    ax2 = fig.add_subplot(122)
    ax2.scatter(x_2dim_PCA[:, 0], x_2dim_PCA[:, 1], c=y, s=n_cluster, label=y)
    ax2.set_xticks([])  # 去掉x轴
    ax2.set_yticks([])  # 去掉y轴
    ax2.set_title("PCA")
    plt.show()

