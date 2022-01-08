# Detection of outliers using Autoencoder  (Deep learning project)
By Yun RU & Xuran HUANG

<img src="https://github.com/ruyunnuyur/Deep-learning-project/blob/6e812364ccad1bd64d5e10f6fbe37bce35ef3dad/1_F_yiILIE954AZPgPADx76A.png" width="500" height="300">

## Introduction
There exists many ways to detect anomaly, One-class SVMs, Elliptic Envelopes... These methods belong to the field of machine learning, however there are also many models for anomaly detection in deep learning area. Autoencoders, a type of unsupervised neural network, are exactly one of the models. In the following sections, we will apply three different autoencoders which are simple autoencoder, deep fully-connected autoencoder and variational autoencoder, to detect the outliers in the datasets that we built based on MNIST data and fashionMNIST data and compare their results.

## How to detect the outliers by using autoencoder?
<img src="autoencoder_schema.jpg" width="600" height="200">
The algorithm of autoencoder is composed by two parts: encoder and decoder. The encoder consists in compressing the inputs into a lower-dimensions space (so-called latent-space representaion) and then the decoder attempts to reconstruct the original data based one the lower-dimensions space. The model of autoencoder is an special type of neural network where the hidden layers have fewer neurons than the input layer and the output layer. This condition forces the hidden layers to extract the most important information from the input and get rid of the noises. The reconstructed images often lose some information compared to the original data, thus we could measure the MSE (Mean Square Error) between them to evaluate the performance of the autoencoder.

&nbsp;

Given the information above, how can autoencoder detect the outliers? Let's take the case of image for example. Imagine that we trained thousands of images of tiger, our autoencoder is familiar with the pictures of tiger and know how to reconsctruct them with the lowest loss. However, if we give the autoencoder an image of elephant, the autoencoder do not know how to recontruct it into an elephant and the output image will look different from the original image, hence we obtain a high MSE for the outliers. By looking for the observations who have a high MSE between the reconstructed image and original image, we will find the outliers!

## Data discription (MNIST & Fashion MNIST)

### MNIST Dataset

<img src="MnistExamples.png" width="500" height="300">

MNIST is a large database containing 70 000 handwritten digits images. 60 000 of them are in the built-in training set and 10 000 are in the built-in test set. Each image is stored as a 28\*28 matrix of pixel. Since we can not train each image as matrix, we flatten the 28\*28 matrix into a vector with 784 elements and we normalize the data. We also possess a label variable which indicates the digit number of the image. In order to adapt to our framework of anormaly detection, we have to rebuild the training set and test set. Before doing so, we want to distinguish two conceptions: novelty detection and anormaly detection. 

**Novelty detection**: The training data consists only of normal observations but the test set contains some new data that the training model has never seen.

**Outlier detection**: The training data consists of both normal data and outliers.

Some papers did the novelty detection and others did the outilier detection. We want to do both of them, so we created 3 data frameworks. The detailed information is as follow:

<img src="table.png" width="600" height="150">

\**The number of anomalies account for 5% of the number of normal data in both training set and test set, the valid label is the label of the normal and the anormal label is the label for the anomaly*

Data framework 1 can be considered as novelty detection, since it contains 0 anomaly in the training set 
Data framework 2 can be considered as outlier detection, there are 337 outliers in the training set. 
Data framework 3 can be considered as novelty detection, but we consider all the labels except 8 as normal data

These 3 dataframes will be applied to the different models of autoencoder.

### Fashion MNIST Dataset

<img src="FashionMinist.jpg" width="500" height="200">

Similar to the MNIST dataset, the Fashion MINIST contains 60 000 images in the training set and 10 000 in the test set and the images are also stored as a 28\*28 matrix. The difference is that the images are changed to clothing images. There are ten types of clothing, their types matched with the lable numbers are as follow:

0 T-shirt/top 1 Trouser 2 Pullover 3 Dress 4 Coat 5 Sandal 6 Shirt 7 Sneaker 8 Bag 9 Ankle boot



## Three methods applied and their performances

**Simple autoencoder**: Simple autoencoder is the simplest model to start with. It is consist of an encoder model with a single fully-connected layer, and a decoder model also with a single fully-connected layer. We use 'sigmoid' as activation function for decoder layer because we want a binary result.

**Deep fully-connected autoencoder**: Instead of using one layer for encoder model and decoder model respectively, we can use several layers to add complexity to the model. More layers allow the model to learn more detailed relationships within the data and how the features interact with each other on a non-linear level.

**Variational autoencoder**: A traditional autoencoder takes in a vector and generates a latent vector (smaller dimension than the initial vecotr) which decoder model can reproduce. But it is possible that the encoder chooses to position two similar data points relatively far from each other in latent space if that minimizes the reconstruction loss. The output of the encoder function in such architecture generates a discrete latent space and often resembles an overfitted model. As a result, although an autoencoder can form a latent space that allows it to be very accurate in its task, there is not much we can assume on the distribution and topology of the latent space it generates, or on how data is organized there. So we have variational autoencoder which generates two vectors which are mean and variance of the latent vector. What's more, variational autoencoder's loss function takes into account a construction loss component which forces the encoder to generate latent features that minimize the reconstruction loss, and a KL loss component, which forces the distribution generated by the encoder to be similar to the prior probability of the input vector (assumed to be normal). So we get a more contiunous and smoother latent space.
                                                                                             -Description refers to 'Hands-on Anomaly Detection with Variational Autoencoders'


## What about other database? Applied on FashionMNIST
## Strengths and Weaknesses
size of train set?
