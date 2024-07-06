# Image Classification using Convolutional Neural Networks

## Introduction

A Project to understand and implement a Image Classification Convolutional Neural Network on the [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

Includes pre-trained model files for quick implementation

## How to use

1. Using the CIFAR10 Dataset

    1. To predict output images using the model trained on CIFAR10 dataset , simply execute ``` python app.py -image [path to image] ```

    
2. To train your own model

    1. Edit the code in app.py to meet your need case and run the file using ``` python app.py -train ```


## Architecture

The Model Consists of 2 Convolutions and 2 MaxPooling Layers connected to 3 Fully Connected Layers for predication, it is implemented in the following way using the RELU activation function and CrossEntropyLoss Function with the Stocastic Gradient Descent Optimizer.

Input Image -> Convolutional Layer 1 -> RELU -> MaxPooling Layer -> Convolutional Layer 2 -> RELU -> MaxPooling Layer -> Flattening 

Flattened Output -> Fully Connected Layer 1 -> RELU -> Fully Connected Layer 2 -> RELU ->Fully Connected Layer 3 -> RELU -> Output

The layers are Structured as Follows:

### Convolutional Layer 1
1. Input Channels : 3 (Coloured Image)
2. Output Channels: 6 (Can be increased for more accuracy , but will take longer to train)
3. Kernel Size: 5 (Feature Map where each pixel represents 5x5 section of input image)
4. Stride: 1 (How many places to move when scanning image)

### Convolutional Layer 2
1. Input Channels : 6 (Coloured Image)
2. Output Channels: 16 (Can be increased for more accuracy , but will take longer to train)
3. Kernel Size: 5 (Feature Map where each pixel represents 5x5 section of input image)
4. Stride: 1 (How many places to move when scanning image)

### MaxPooling Layer
1. Kernel Size: 2 (maximum value of 2x2 square is calculated)
2. Stride: 2 (How many places to move when scanning image)

### Fully Connected Layer 1
1. In-Features: 16x5x5 (16 output channel of Convolutional Layer 1 multiplied by 5x5 Kernel Size/Feature Map Information , Flattening Step to be done before this step)
2. Out-Features: 128

### Fully Connected Layer 2
1. In-Features: 128
2. Out-Features: 84

### Fully Connected Layer 3
1. In-Features: 84
2. Out-Features: 10

### ReLU Activation Function

ReLU(x)=max(0,x)

ReLU is a simple yet effective activation function used in deep learning due to its computational efficiency, non-linearity, and ability to mitigate the vanishing gradient problem. It's commonly used in hidden layers of neural networks to introduce non-linearities and enable the network to learn complex mappings from input to output.

### CrossEntropyLoss Function

Cross-Entropy Loss measures the performance of a classification model whose output is a probability value between 0 and 1. It compares this output probability distribution with the actual probability distribution (one-hot encoded labels in the case of classification).

Mathematically, for a single example with CC classes, the Cross-Entropy Loss LL is computed as:

L=−∑yilog⁡(pi)

Where:

    yi​ is the true label (0 or 1) for class i.
    pi​ is the predicted probability that the example belongs to class i.

### Stocastic Gradient Descent

SGD is a popular optimization algorithm used in training machine learning models, especially deep neural networks. It is a variant of Gradient Descent that updates the model parameters iteratively based on the gradients of the loss function computed on small subsets of the training data (mini-batches) rather than the entire dataset.

It Aims to minimize the loss function , by starting from a random point and going till it reaches the lowest point of the loss function for that value


## License 

[MIT](https://test)