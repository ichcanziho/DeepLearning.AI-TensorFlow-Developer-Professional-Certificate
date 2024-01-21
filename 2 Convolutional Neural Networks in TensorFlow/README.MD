# Convolutional Neural Networks in TensorFlow

![cover.png](cover.png)

[curso](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow)

En este curso, aprenderás:

- Manejar datos de imágenes del mundo real

- Pérdida de parcelas y precisión

- Explorar estrategias para evitar el sobreajuste, incluidos el aumento y el abandono

- Aprenda el aprendizaje por transferencia y cómo pueden extraerse de los modelos las características aprendidas

Si usted es un desarrollador de software que quiere construir algoritmos escalables impulsados por IA, necesita entender cómo utilizar las herramientas para construirlos. Este curso forma parte de la próxima Especialización en Aprendizaje Automático en Tensorflow y le enseñará las mejores prácticas para utilizar TensorFlow, un popular marco de trabajo de código abierto para el aprendizaje automático. En el Curso 2 de la Especialización en TensorFlow deeplearning.ai, aprenderá técnicas avanzadas para mejorar el modelo de visión por ordenador que construyó en el Curso 1. Explorará cómo trabajar con imágenes del mundo real de diferentes formas y tamaños, visualizará el recorrido de una imagen a través de convoluciones para entender cómo un ordenador "ve" la información, trazará la pérdida y la precisión, y explorará estrategias para evitar el sobreajuste, incluyendo el aumento y el abandono. Por último, el curso 2 le introducirá en el aprendizaje por transferencia y en cómo pueden extraerse de los modelos las características aprendidas. 

El curso de Aprendizaje Automático y la Especialización en Aprendizaje Profundo de Andrew Ng enseñan los principios más importantes y fundacionales del Aprendizaje Automático y el Aprendizaje Profundo. Esta nueva Especialización en TensorFlow de deeplearning.ai le enseña cómo utilizar TensorFlow para implementar esos principios, de modo que pueda empezar a construir y aplicar modelos escalables a problemas del mundo real. Para desarrollar una comprensión más profunda de cómo funcionan las redes neuronales, le recomendamos que realice la Especialización en Aprendizaje Profundo.

## INDEX 0

- [Larger Dataset](#larger-dataset)
- [Weekly Assignment - Attempt cats vs dogs kaggle challenge](#weekly-assignment---attempt-cats-vs-dogs-kaggle-challenge)
- [Augmentation](#augmentation)
- [Weekly Assignment - Full cats vs dogs using augmentation](#weekly-assignment---full-cats-vs-dogs-using-augmentation)
- [Transfer learning](#transfer-learning)
- [Weekly Assignment - Transfer Learning Horses vs Humans](#weekly-assignment---transfer-learning-horses-vs-humans)
- [Muliclass Classifications](#muliclass-classifications)
- [Weekly Assignment - Multiclass Classification](#weekly-assignment---multiclass-classification)


## Larger Dataset
[<- Return to INDEX 0](#index-0)

En el primer curso de esta especialización, usted tuvo una introducción a TensorFlow, y cómo, con sus APIs de alto nivel usted podría hacer clasificación básica de imágenes, y aprendió un poco sobre Redes Neuronales Convolucionales (ConvNets). En este curso profundizará en el uso de las ConvNets con datos del mundo real, y aprenderá las técnicas que puede utilizar para mejorar el rendimiento de sus ConvNet, especialmente cuando realice clasificación de imágenes¡ En la Semana 1, esta semana, comenzará examinando un conjunto de datos mucho mayor que el que ha estado utilizando hasta ahora: El conjunto de datos Gatos y Perros, ¡que había sido un Desafío Kaggle en clasificación de imágenes!

### Objetivos de aprendizaje

- Conozca las utilidades de Keras para el preprocesamiento de datos de imagen, en particular la clase ImageDataGenerator
- Desarrollar funciones de ayuda para mover archivos por el sistema de archivos de forma que puedan ser alimentados al ImageDataGenerator
- Aprenda a trazar las precisiones de entrenamiento y validación para evaluar el rendimiento del modelo
- Construir un clasificador utilizando redes neuronales convolucionales para realizar una clasificación de gatos frente a perros


### INDEX 1

- [Introduction, A conversation with Andrew Ng](#introduction-a-conversation-with-andrew-ng)
- [Where to find the notebooks for this course](#where-to-find-the-notebooks-for-this-course)
- [A conversation with Andrew Ng 1](#a-conversation-with-andrew-ng-1)
- [The cats vs dogs dataset](#the-cats-vs-dogs-dataset)
- [Training with the cats vs. dogs dataset](#training-with-the-cats-vs-dogs-dataset)
- [Looking at the notebook (Lab 1)](#looking-at-the-notebook-lab-1)
- [Have questions, issues or ideas? Join our Community!](#have-questions-issues-or-ideas-join-our-community)
- [Working through the notebook](#working-through-the-notebook)
- [What you'll see next](#what-youll-see-next)
- [Fixing through cropping](#fixing-through-cropping)
- [Visualizing the effect of the convolutions](#visualizing-the-effect-of-the-convolutions)
- [Looking at accuracy and loss](#looking-at-accuracy-and-loss)
- [What have we seen so far?](#what-have-we-seen-so-far)
- [Week 1 Quiz](#week-1-quiz)
- [Week 1 Wrap up](#week-1-wrap-up)
- [Lecture Notes Week 1](#lecture-notes-week-1)

### Introduction, A conversation with Andrew Ng
[<- Return to INDEX 1](#index-1)

### Where to find the notebooks for this course
[<- Return to INDEX 1](#index-1)

### A conversation with Andrew Ng 1
[<- Return to INDEX 1](#index-1)

### The cats vs dogs dataset
[<- Return to INDEX 1](#index-1)

### Training with the cats vs. dogs dataset
[<- Return to INDEX 1](#index-1)

### Looking at the notebook (Lab 1)
[<- Return to INDEX 1](#index-1)

### Have questions, issues or ideas? Join our Community!
[<- Return to INDEX 1](#index-1)

### Working through the notebook
[<- Return to INDEX 1](#index-1)

### What you'll see next
[<- Return to INDEX 1](#index-1)

### Fixing through cropping
[<- Return to INDEX 1](#index-1)

### Visualizing the effect of the convolutions
[<- Return to INDEX 1](#index-1)

### Looking at accuracy and loss
[<- Return to INDEX 1](#index-1)

### What have we seen so far?
[<- Return to INDEX 1](#index-1)

### Week 1 Quiz
[<- Return to INDEX 1](#index-1)

### Week 1 Wrap up
[<- Return to INDEX 1](#index-1)

### Lecture Notes Week 1
[<- Return to INDEX 1](#index-1)

## Weekly Assignment - Attempt cats vs dogs kaggle challenge
[<- Return to INDEX 0](#index-0)

### Assignment Troubleshooting Tips

### Programming Assignment: Cats vs Dogs

## Augmentation
[<- Return to INDEX 0](#index-0)

Hasta este punto, habrá oído muchas veces el término sobreadaptación. La sobreadaptación es simplemente el concepto de estar demasiado especializado en el entrenamiento, es decir, que su modelo es muy bueno clasificando aquello para lo que ha sido entrenado, pero no tan bueno clasificando cosas que no ha visto. Para generalizar su modelo de forma más eficaz, necesitará por supuesto una mayor amplitud de muestras con las que entrenarlo. Eso no siempre es posible, pero un buen atajo potencial para ello es el Aumento de la imagen, con el que se ajusta el conjunto de entrenamiento para aumentar potencialmente la diversidad de temas que abarca. ¡Aprenderá todo sobre ello esta semana!

### Objetivos de aprendizaje

- Reconocer el impacto de añadir el aumento de imágenes al proceso de formación, especialmente en el tiempo
- Demuestre la sobreadaptación o la falta de ella trazando las precisiones de entrenamiento y validación
- Familiarizarse con los parámetros del ImageDataGenerator utilizados para llevar a cabo el aumento de imágenes
- Aprenda a mitigar el sobreajuste utilizando técnicas de aumento de datos

### INDEX 2

- [A conversation with Andrew Ng 2](#a-conversation-with-andrew-ng-2)
- [Image Augmentation](#image-augmentation)
- [Introducing augmentation](#introducing-augmentation)
- [Start Coding...](#start-coding)
- [Coding augmentation with ImageDataGenerator](#coding-augmentation-with-imagedatagenerator)
- [Looking at the notebook (Lab 1)](#looking-at-the-notebook-lab-1)
- [Demonstrating overfitting in cats vs dogs](#demonstrating-overfitting-in-cats-vs-dogs)
- [The impact of augmentation on Cats vs. Dogs](#the-impact-of-augmentation-on-cats-vs-dogs)
- [Adding augmentation to cats vs dogs](#adding-augmentation-to-cats-vs-dogs)
- [Image Augmentation with Horses vs Humans! (Lab 2)](#image-augmentation-with-horses-vs-humans-lab-2)
- [Exploring augmentation with horses vs humans](#exploring-augmentation-with-horses-vs-humans)
- [What have we seen so far? 2](#what-have-we-seen-so-far-2)
- [Week 2 Quiz](#week-2-quiz)
- [Week 2 Wrap up](#week-2-wrap-up)
- [Lecture Notes Week 2](#lecture-notes-week-2)

### A conversation with Andrew Ng 2
[<- Return to INDEX 2](#index-2)

### Image Augmentation
[<- Return to INDEX 2](#index-2)

### Introducing augmentation
[<- Return to INDEX 2](#index-2)

### Start Coding...
[<- Return to INDEX 2](#index-2)

### Coding augmentation with ImageDataGenerator
[<- Return to INDEX 2](#index-2)

### Looking at the notebook (Lab 1)
[<- Return to INDEX 2](#index-2)

### Demonstrating overfitting in cats vs dogs
[<- Return to INDEX 2](#index-2)

### The impact of augmentation on Cats vs. Dogs
[<- Return to INDEX 2](#index-2)

### Adding augmentation to cats vs dogs
[<- Return to INDEX 2](#index-2)

### Image Augmentation with Horses vs Humans! (Lab 2)
[<- Return to INDEX 2](#index-2)

### Exploring augmentation with horses vs humans
[<- Return to INDEX 2](#index-2)

### What have we seen so far? 2
[<- Return to INDEX 2](#index-2)

### Week 2 Quiz
[<- Return to INDEX 2](#index-2)

### Week 2 Wrap up
[<- Return to INDEX 2](#index-2)

### Lecture Notes Week 2
[<- Return to INDEX 2](#index-2)

## Weekly Assignment - Full cats vs dogs using augmentation
[<- Return to INDEX 0](#index-0)

### Cats vs Dogs with Data Augmentation

## Transfer learning
[<- Return to INDEX 0](#index-0)

Construir modelos para usted mismo está muy bien y puede ser muy potente. Pero, como ha visto, puede verse limitado por los datos que tenga a mano. No todo el mundo tiene acceso a conjuntos de datos masivos o a la potencia de cálculo necesaria para entrenarlos con eficacia. El aprendizaje por transferencia puede ayudar a resolver esto -- donde personas con modelos entrenados en grandes conjuntos de datos los entrenan, para que usted pueda utilizarlos directamente, o bien, puede utilizar las características que ellos han aprendido y aplicarlas a su escenario. Esto es el aprendizaje por transferencia, ¡y lo estudiará esta semana!

### Objetivos de aprendizaje

- Domine el tipo de capa keras conocido como dropout para evitar el sobreajuste
- Logre el aprendizaje por transferencia en código utilizando la API keras
- Codifique un modelo que implemente la API funcional de Keras en lugar del modelo secuencial comúnmente utilizado
- Aprenda a congelar capas de un modelo existente para aplicar con éxito el aprendizaje por transferencia
- Explore el concepto de aprendizaje por transferencia para utilizar las convoluciones aprendidas por un modelo diferente a partir de un conjunto de datos mayor

### INDEX 3

- [A conversation with Andrew Ng 3](#a-conversation-with-andrew-ng-3)
- [Understanding transfer learning: the concepts](#understanding-transfer-learning-the-concepts)
- [Coding transfer learning from the inception model](#coding-transfer-learning-from-the-inception-model)
- [Adding your DNN](#adding-your-dnn)
- [Coding your own model with transferred features](#coding-your-own-model-with-transferred-features)
- [Using dropout](#using-dropout)
- [Exploring dropouts](#exploring-dropouts)
- [Applying Transfer Learning to Cats v Dogs (Lab 1)](#applying-transfer-learning-to-cats-v-dogs-lab-1)
- [Exploring Transfer Learning with Inception](#exploring-transfer-learning-with-inception)
- [What have we seen so far? 3](#what-have-we-seen-so-far-3)
- [Week 3 Quiz](#week-3-quiz)
- [Week 3 Wrap up](#week-3-wrap-up)
- [Lecture Notes Week 3](#lecture-notes-week-3)

### A conversation with Andrew Ng 3
[<- Return to INDEX 3](#index-3)

### Understanding transfer learning: the concepts
[<- Return to INDEX 3](#index-3)

### Coding transfer learning from the inception model
[<- Return to INDEX 3](#index-3)

### Adding your DNN
[<- Return to INDEX 3](#index-3)

### Coding your own model with transferred features
[<- Return to INDEX 3](#index-3)

### Using dropout
[<- Return to INDEX 3](#index-3)

### Exploring dropouts
[<- Return to INDEX 3](#index-3)

### Applying Transfer Learning to Cats v Dogs (Lab 1)
[<- Return to INDEX 3](#index-3)

### Exploring Transfer Learning with Inception
[<- Return to INDEX 3](#index-3)

### What have we seen so far? 3
[<- Return to INDEX 3](#index-3)

### Week 3 Quiz
[<- Return to INDEX 3](#index-3)

### Week 3 Wrap up
[<- Return to INDEX 3](#index-3)

### Lecture Notes Week 3
[<- Return to INDEX 3](#index-3)

## Weekly Assignment - Transfer Learning Horses vs Humans
[<- Return to INDEX 0](#index-0)

### Transfer Learning - Horses vs Humans

## Muliclass Classifications
[<- Return to INDEX 0](#index-0)

Ha recorrido un largo camino, ¡enhorabuena! Queda una cosa por hacer antes de pasar de las ConvNets al siguiente módulo, y es ir más allá de la clasificación binaria. Cada uno de los ejemplos que ha hecho hasta ahora implicaba clasificar una cosa u otra: caballo o humano, gato o perro. Al pasar de la clasificación binaria a la categórica hay algunas consideraciones de codificación que debe tener en cuenta. ¡Las verá esta semana!

### Objetivos de aprendizaje

- Construir un clasificador multiclase para el conjunto de datos MNIST del lenguaje de signos
- Aprenda a configurar correctamente los parámetros del ImageDataGenerator y las funciones de definición del modelo para la clasificación multiclase
- Comprender la diferencia entre utilizar archivos de imagen reales frente a imágenes codificadas en otros formatos y cómo esto cambia los métodos disponibles al utilizar ImageDataGenerator
- Codifique una función de ayuda para analizar un archivo CSV sin procesar que contenga la información de los valores de píxel de las imágenes utilizadas

### INDEX 4

- [A conversation with Andrew Ng 4](#a-conversation-with-andrew-ng-4)
- [Moving from binary to multi-class classification](#moving-from-binary-to-multi-class-classification)
- [Introducing the Rock-Paper-Scissors dataset](#introducing-the-rock-paper-scissors-dataset)
- [Explore multi-class with Rock Paper Scissors dataset](#explore-multi-class-with-rock-paper-scissors-dataset)
- [Check out the code! (Lab 1)](#check-out-the-code-lab-1)
- [Train a classifier with Rock Paper Scissors](#train-a-classifier-with-rock-paper-scissors)
- [Try testing the classifier](#try-testing-the-classifier)
- [Test the Rock Paper Scissors classifier](#test-the-rock-paper-scissors-classifier)
- [What have we seen so far? 4](#what-have-we-seen-so-far-4)
- [Week 4 Quiz](#week-4-quiz)
- [Lecture Notes Week 4](#lecture-notes-week-4)
- [IMPORTANT Reminder about end of access to Lab Notebooks](#important-reminder-about-end-of-access-to-lab-notebooks)
- [Wrap up 4](#wrap-up-4)
- [A conversation with Andrew Ng 5](#a-conversation-with-andrew-ng-5)
- [Acknowledgments](#acknowledgments)

### A conversation with Andrew Ng 4
[<- Return to INDEX 4](#index-4)

### Moving from binary to multi-class classification
[<- Return to INDEX 4](#index-4)

### Introducing the Rock-Paper-Scissors dataset
[<- Return to INDEX 4](#index-4)

### Explore multi-class with Rock Paper Scissors dataset
[<- Return to INDEX 4](#index-4)

### Check out the code! (Lab 1)
[<- Return to INDEX 4](#index-4)

### Train a classifier with Rock Paper Scissors
[<- Return to INDEX 4](#index-4)

### Try testing the classifier
[<- Return to INDEX 4](#index-4)

### Test the Rock Paper Scissors classifier
[<- Return to INDEX 4](#index-4)

### What have we seen so far? 4
[<- Return to INDEX 4](#index-4)

### Week 4 Quiz
[<- Return to INDEX 4](#index-4)

### Lecture Notes Week 4
[<- Return to INDEX 4](#index-4)

### IMPORTANT Reminder about end of access to Lab Notebooks
[<- Return to INDEX 4](#index-4)

### Wrap up 4
[<- Return to INDEX 4](#index-4)

### A conversation with Andrew Ng 5
[<- Return to INDEX 4](#index-4)

### Acknowledgments
[<- Return to INDEX 4](#index-4)

## Weekly Assignment - Multiclass Classification
[<- Return to INDEX 0](#index-0)

### Classification: Beyond two classes

