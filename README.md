**CNN for CIFAR-10 Image Classification with PyTorch**
This project implements a Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset using the PyTorch deep learning framework. The model is designed to achieve high accuracy in image classification tasks by leveraging convolutional layers, pooling, and fully connected layers.

**Features**
CIFAR-10 Dataset: A dataset of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
CNN Model: A deep learning model consisting of convolutional layers, activation functions (ReLU), max-pooling, and fully connected layers.
Training & Evaluation: The model is trained using the cross-entropy loss function and Adam optimizer, with evaluation on the test set.
PyTorch Implementation: Fully implemented using PyTorch, leveraging its flexibility and ease of use for deep learning tasks.
Requirements
Python 3.x
PyTorch
torchvision
matplotlib
numpy

**Model Architecture**
Convolutional Layers: Layers with filters to extract features from the images.
ReLU Activation: Used to introduce non-linearity into the model.
Max Pooling: To reduce spatial dimensions and prevent overfitting.
Fully Connected Layers: For classification based on the extracted features.
Results
The model's performance can be evaluated by accuracy on the test set after training. The accuracy should increase as the number of epochs grows.
