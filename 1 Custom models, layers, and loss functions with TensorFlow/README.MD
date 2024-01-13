# Custom Models, Layers, and Loss Functions with TensorFlow

En este curso, aprenderás:

- Comparar las APIs Funcional y Secuencial, descubrir nuevos modelos que puedes construir con la API Funcional y crear un modelo que produzca múltiples salidas, incluida una red Siamesa.
- Construir funciones de pérdida personalizadas (incluida la función de pérdida contrastante utilizada en una red Siamesa) para medir qué tan bien está funcionando un modelo y ayudar a tu red neuronal a aprender de los datos de entrenamiento.
- Crear capas personalizadas a partir de capas estándar existentes para tus modelos, personalizar una capa de red con una capa lambda, comprender las diferencias entre ellas, aprender qué compone una capa personalizada y explorar funciones de activación.
- Ampliar modelos existentes para agregar funcionalidad personalizada, aprender a definir tu propia clase personalizada en lugar de usar las APIs Funcional o Secuencial, construir modelos que pueden heredarse de la clase Model de TensorFlow y crear una red residual (ResNet) mediante la definición de una clase de modelo personalizada.

**La Especialización en TensorFlow**: Técnicas Avanzadas de DeepLearning. AI introduce las características de TensorFlow que brindan a los aprendices más control sobre la arquitectura de sus modelos y herramientas que les ayudan a crear y entrenar modelos avanzados de aprendizaje automático.

Esta especialización está diseñada para ingenieros de software y aprendizaje automático en etapas tempranas y medias de su carrera, con un entendimiento fundamental de TensorFlow que buscan ampliar sus conocimientos y habilidades aprendiendo características avanzadas de TensorFlow para construir modelos potentes.

## INDEX 0

- [A new programming paradigm](#a-new-programming-paradigm)
- [Weekly Assignment - Your first Neural Network](#weekly-assignment---your-first-neural-network)
- [Introduction to Computer Vision](#introduction-to-computer-vision)
- [Weekly Assignment - Implement a Deep Neural Network to recognize handwritten digits](#weekly-assignment---implement-a-deep-neural-network-to-recognize-handwritten-digits)
- [Enhancing Vision with Convolutional Neural Networks](#enhancing-vision-with-convolutional-neural-networks)
- [Weekly Assignment - Improving DNN Performance using Convolutions](#weekly-assignment---improving-dnn-performance-using-convolutions)
- [Using Real-world Images](#using-real-world-images)
- [Weekly Assignment - Handling Complex Images](#weekly-assignment---handling-complex-images)

## A new programming paradigm
[<- Return to INDEX 0](#index-0)

### INDEX 1

- [Introduction: A conversation with Andrew Ng](#introduction-a-conversation-with-andrew-ng)
- [Where to find the notebooks for this course](#where-to-find-the-notebooks-for-this-course)
- [A primer in machine learning](#a-primer-in-machine-learning)
- [The Hello World of neural networks](#the-hello-world-of-neural-networks)
- [Intake Survey](#intake-survey)
- [Join our community](#join-our-community)
- [From rules to data](#from-rules-to-data)
- [Working through Hello World in TensorFlow and Python](#working-through-hello-world-in-tensorflow-and-python)
- [Get started with Google Colaboratory (Coding TensorFlow)](#get-started-with-google-colaboratory-coding-tensorflow)
- [Faqs about Google Colab](#faqs-about-google-colab)
- [Try it for yourself (Lab 1)](#try-it-for-yourself-lab-1)
- [Week 1 Quiz](#week-1-quiz)
- [Lecture Notes 1](#lecture-notes-1)

### Introduction: A conversation with Andrew Ng
[<- Return to INDEX 1](#index-1)

### Where to find the notebooks for this course
[<- Return to INDEX 1](#index-1)

### A primer in machine learning
[<- Return to INDEX 1](#index-1)

### The Hello World of neural networks
[<- Return to INDEX 1](#index-1)

### Intake Survey
[<- Return to INDEX 1](#index-1)

### Join our community
[<- Return to INDEX 1](#index-1)

### From rules to data
[<- Return to INDEX 1](#index-1)

### Working through Hello World in TensorFlow and Python
[<- Return to INDEX 1](#index-1)

### Get started with Google Colaboratory (Coding TensorFlow)
[<- Return to INDEX 1](#index-1)

### Faqs about Google Colab
[<- Return to INDEX 1](#index-1)

### Try it for yourself (Lab 1)
[<- Return to INDEX 1](#index-1)

### Week 1 Quiz
[<- Return to INDEX 1](#index-1)

### Lecture Notes 1
[<- Return to INDEX 1](#index-1)

## Weekly Assignment - Your first Neural Network
[<- Return to INDEX 0](#index-0)

### INDEX 3

- [Assignment Troubleshooting Tips](#assignment-troubleshooting-tips)
- [Downloading your Notebook and refreshing your workspace](#downloading-your-notebook-and-refreshing-your-workspace)
- [Housing Prices](#housing-prices)
- [Week 1 Resources](#week-1-resources)

### Assignment Troubleshooting Tips
[<- Return to INDEX 3](#index-3)

### Downloading your Notebook and refreshing your workspace
[<- Return to INDEX 3](#index-3)

### Housing Prices
[<- Return to INDEX 3](#index-3)

### Week 1 Resources
[<- Return to INDEX 3](#index-3)

## Introduction to Computer Vision
[<- Return to INDEX 0](#index-0)

### INDEX 4

- [A conversation with Andrew Ng](#a-conversation-with-andrew-ng)
- [An Introduction to computer vision](#an-introduction-to-computer-vision)
- [Exploring how to use data](#exploring-how-to-use-data)
- [Writing code to load training data](#writing-code-to-load-training-data)
- [The structure of Fashion MNIST data](#the-structure-of-fashion-mnist-data)
- [Coding a Computer Vision Neural Network](#coding-a-computer-vision-neural-network)
- [See how it's done](#see-how-its-done)
- [Walk through a Notebook for computer vision](#walk-through-a-notebook-for-computer-vision)
- [Get Hands-on with computer vision (Lab 1)](#get-hands-on-with-computer-vision-lab-1)
- [Using Callbacks to control training](#using-callbacks-to-control-training)
- [See how to implement Callbacks (Lab 2)](#see-how-to-implement-callbacks-lab-2)
- [Walk through a notebook with Callbacks](#walk-through-a-notebook-with-callbacks)
- [Week 2 Quiz](#week-2-quiz)
- [Lecture Notes Week 2](#lecture-notes-week-2)

### A conversation with Andrew Ng
[<- Return to INDEX 4](#index-4)

### An Introduction to computer vision
[<- Return to INDEX 4](#index-4)

### Exploring how to use data
[<- Return to INDEX 4](#index-4)

### Writing code to load training data
[<- Return to INDEX 4](#index-4)

### The structure of Fashion MNIST data
[<- Return to INDEX 4](#index-4)

### Coding a Computer Vision Neural Network
[<- Return to INDEX 4](#index-4)

### See how it's done
[<- Return to INDEX 4](#index-4)

### Walk through a Notebook for computer vision
[<- Return to INDEX 4](#index-4)

### Get Hands-on with computer vision (Lab 1)
[<- Return to INDEX 4](#index-4)

### Using Callbacks to control training
[<- Return to INDEX 4](#index-4)

### See how to implement Callbacks (Lab 2)
[<- Return to INDEX 4](#index-4)

### Walk through a notebook with Callbacks
[<- Return to INDEX 4](#index-4)

### Week 2 Quiz
[<- Return to INDEX 4](#index-4)

### Lecture Notes Week 2
[<- Return to INDEX 4](#index-4)

## Weekly Assignment - Implement a Deep Neural Network to recognize handwritten digits
[<- Return to INDEX 0](#index-0)

### INDEX 5

- [Implementing Callbacks in TensorFlow using the MNIST Dataset](#implementing-callbacks-in-tensorflow-using-the-mnist-dataset)

### Implementing Callbacks in TensorFlow using the MNIST Dataset
[<- Return to INDEX 5](#index-5)

## Enhancing Vision with Convolutional Neural Networks
[<- Return to INDEX 0](#index-0)

### INDEX 6

- [A conversation with Andre Ng 3](#a-conversation-with-andre-ng-3)
- [What are convolution and pooling](#what-are-convolution-and-pooling)
- [Coding convolutions and pooling layers](#coding-convolutions-and-pooling-layers)
- [Implementing convolutional layers](#implementing-convolutional-layers)
- [Learn more about convolutions](#learn-more-about-convolutions)
- [Implementing pooling layers](#implementing-pooling-layers)
- [Getting hands-on, your first ConvNet](#getting-hands-on-your-first-convnet)
- [Improving the Fashion classifier with convolutions](#improving-the-fashion-classifier-with-convolutions)
- [Try it for yourself (Lab 1 week 3)](#try-it-for-yourself-lab-1-week-3)
- [Walking through convolutions](#walking-through-convolutions)
- [Experiment with filters and pools (Lab 2 week 3)](#experiment-with-filters-and-pools-lab-2-week-3)
- [Week 3 Quiz](#week-3-quiz)
- [Lecture Notes Week 3](#lecture-notes-week-3)

### A conversation with Andre Ng 3
[<- Return to INDEX 6](#index-6)

### What are convolution and pooling
[<- Return to INDEX 6](#index-6)

### Coding convolutions and pooling layers
[<- Return to INDEX 6](#index-6)

### Implementing convolutional layers
[<- Return to INDEX 6](#index-6)

### Learn more about convolutions
[<- Return to INDEX 6](#index-6)

### Implementing pooling layers
[<- Return to INDEX 6](#index-6)

### Getting hands-on, your first ConvNet
[<- Return to INDEX 6](#index-6)

### Improving the Fashion classifier with convolutions
[<- Return to INDEX 6](#index-6)

### Try it for yourself (Lab 1 week 3)
[<- Return to INDEX 6](#index-6)

### Walking through convolutions
[<- Return to INDEX 6](#index-6)

### Experiment with filters and pools (Lab 2 week 3)
[<- Return to INDEX 6](#index-6)

### Week 3 Quiz
[<- Return to INDEX 6](#index-6)

### Lecture Notes Week 3
[<- Return to INDEX 6](#index-6)

## Weekly Assignment - Improving DNN Performance using Convolutions
[<- Return to INDEX 0](#index-0)

### INDEX 7

- [Improve MNIST with convolutions](#improve-mnist-with-convolutions)

### Improve MNIST with convolutions
[<- Return to INDEX 7](#index-7)

## Using Real-world Images
[<- Return to INDEX 0](#index-0)

### INDEX 8

- [A conversation with Andrew Ng Week 4](#a-conversation-with-andrew-ng-week-4)
- [Explore an impactful, real-world solution](#explore-an-impactful-real-world-solution)
- [Understanding ImageDataGenerator](#understanding-imagedatagenerator)
- [Designing the neural network](#designing-the-neural-network)
- [Defining a ConvNet to use complex images](#defining-a-convnet-to-use-complex-images)
- [Train the ConvNet with ImageDataGenerator](#train-the-convnet-with-imagedatagenerator)
- [Training the ConvNet](#training-the-convnet)
- [Exploring the solution](#exploring-the-solution)
- [Walking through developing a ConvNet](#walking-through-developing-a-convnet)
- [Training the neural network](#training-the-neural-network)
- [Walking through training the ConvNet](#walking-through-training-the-convnet)
- [Experiment with the horse or human classifier (Lab 1)](#experiment-with-the-horse-or-human-classifier-lab-1)
- [Adding automatic validation to test accuracy](#adding-automatic-validation-to-test-accuracy)
- [Get hands-on and use validation (Lab 2)](#get-hands-on-and-use-validation-lab-2)
- [Exploring the impact of compressing images](#exploring-the-impact-of-compressing-images)
- [Get Hands-on with compacted images (Lab 3)](#get-hands-on-with-compacted-images-lab-3)
- [Week 4 Quiz](#week-4-quiz)
- [Lecture Notes Week 4](#lecture-notes-week-4)

### A conversation with Andrew Ng Week 4
[<- Return to INDEX 8](#index-8)

### Explore an impactful, real-world solution
[<- Return to INDEX 8](#index-8)

### Understanding ImageDataGenerator
[<- Return to INDEX 8](#index-8)

### Designing the neural network
[<- Return to INDEX 8](#index-8)

### Defining a ConvNet to use complex images
[<- Return to INDEX 8](#index-8)

### Train the ConvNet with ImageDataGenerator
[<- Return to INDEX 8](#index-8)

### Training the ConvNet
[<- Return to INDEX 8](#index-8)

### Exploring the solution
[<- Return to INDEX 8](#index-8)

### Walking through developing a ConvNet
[<- Return to INDEX 8](#index-8)

### Training the neural network
[<- Return to INDEX 8](#index-8)

### Walking through training the ConvNet
[<- Return to INDEX 8](#index-8)

### Experiment with the horse or human classifier (Lab 1)
[<- Return to INDEX 8](#index-8)

### Adding automatic validation to test accuracy
[<- Return to INDEX 8](#index-8)

### Get hands-on and use validation (Lab 2)
[<- Return to INDEX 8](#index-8)

### Exploring the impact of compressing images
[<- Return to INDEX 8](#index-8)

### Get Hands-on with compacted images (Lab 3)
[<- Return to INDEX 8](#index-8)

### Week 4 Quiz
[<- Return to INDEX 8](#index-8)

### Lecture Notes Week 4
[<- Return to INDEX 8](#index-8)

## Weekly Assignment - Handling Complex Images
[<- Return to INDEX 0](#index-0)

### INDEX 9

- [Reminder about the end of access to Lab Notebooks](#reminder-about-the-end-of-access-to-lab-notebooks)
- [Handling complex images](#handling-complex-images)
- [Wrap up](#wrap-up)
- [A conversation with Andrew Final](#a-conversation-with-andrew-final)
- [Acknowledgments](#acknowledgments)

### Reminder about the end of access to Lab Notebooks
[<- Return to INDEX 9](#index-9)

### Handling complex images
[<- Return to INDEX 9](#index-9)

### Wrap up
[<- Return to INDEX 9](#index-9)

### A conversation with Andrew Final
[<- Return to INDEX 9](#index-9)

### Acknowledgments
[<- Return to INDEX 9](#index-9)