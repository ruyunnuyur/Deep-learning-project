# Detection of outliers using Autoencoder  (Deep learning project)
By Yun RU & Xuran HUANG

<img src="https://github.com/ruyunnuyur/Deep-learning-project/blob/6e812364ccad1bd64d5e10f6fbe37bce35ef3dad/1_F_yiILIE954AZPgPADx76A.png" width="500" height="300">

## Introduction
There exists many ways to detect anomaly, One-class SVMs, Elliptic Envelopes... These methods belong to the field of machine learning, however there are also many models for anomaly detection in deep learning area. Autoencoders, a type of unsupervised neural network, are exactly one of the models. In the following sections, we will apply three different autoencoders which are simple autoencoder, deep fully-connected autoencoder and variational autoencoder, to detect the outliers in the datasets that we built based on MNIST data and fashionMNIST data and compare their results.

## How to detect the outliers by using autoencoder
<img src="autoencoder_schema.jpg" width="600" height="200">
The algorithm of autoencoder is composed by two parts: encoder and decoder. The encoder consists in compressing the inputs into a lower-dimensions space (so-called latent-space representaion) and then the decoder attempts to reconstruct the original data based one the lower-dimensions space. The model of autoencoder is an special type of neural network where the hidden layers have fewer neurons than the input layer and the output layer. This condition forces the hidden layers to extract the most important information from the input and get rid of the noises. The reconstructed images often lose some information compared to the original data, thus we could measure the MSE (Mean Square Error) between them to evaluate the performance of the autoencoder.


Given the information above, how can autoencoder detect the outliers? Let's take the case of image for example. Imagine that we trained thousands of images of tiger, our autoencoder is familiar with the pictures of tiger and know how to reconsctruct them with the lowest loss. However, if we give the autoencoder an image of elephant, the autoencoder do not know how to recontruct it into an elephant and the output image will look different from the original image, hence we obtain a high MSE for the outliers. By looking for the observations who have a high MSE between the reconstructed image and original image, we will find the outliers!

## Data discription (MNIST)
Novelty detection v.s. anomaly detection How we build our train set and test set
## Three methods applied and their performances
## Applied on another dataset (FashionMNIST)
## Strengths and Weaknesses
size of train set?
