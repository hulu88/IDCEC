# IDCEC
# Improved deep convolutional embedded clustering with re-selectable sample training 


This is repository contains the code for the paper ( Pattern Recognition, 2022)

Improved deep convolutional embedded clustering with re-selectable sample training,<br>
Hu Lu and Chao Chen* 

## Abstract
The deep clustering algorithm can learn the latent features of the embedded subspace, and further realize the clustering of samples in the feature space. The existing deep clustering algorithms mostly integrate neural networks and traditional clustering algorithms. However, for sample sets with many noise points, the effect of the clustering remains unsatisfactory. To address this issue, we propose an improved deep convolutional embedded clustering algorithm using reliable samples (IDCEC) in this paper. The algorithm first uses the convolutional autoencoder to extract features and cluster the samples. Then we select reli- able samples with pseudo-labels and pass them to the convolutional neural network for training to get a better clustering model. We construct a new loss function for backpropagation training and implement an unsupervised deep clustering method. To verify the performance of the method proposed in this paper, we conducted experimental tests on standard data sets such as MNIST and USPS. Experimental results show that our method has better performance compared to traditional clustering algorithms and the state-of-the-art deep clustering algorithm under four clustering metrics.

## Platform

This code was developed and tested with: 
h5py==2.10.0
scikit-learn==0.22
tensorflow-gpu==1.15.0
keras==2.1.6



## Citation 

If you use this code for your research, please cite our paper:
-------
@article{IDCEC,<br>
title={ Improved deep convolutional embedded clustering with re-selectable sample training },<br>
  author={Hu Lu, Chao Chen, Hui Wei, Zhongchen Ma, Ke Jiang and Yingquan Wang },<br>
  journal={Pattern Recognition},<br>
  year={2022},<br>
}
