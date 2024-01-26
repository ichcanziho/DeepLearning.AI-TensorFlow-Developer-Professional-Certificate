# Natural Language Processing in TensorFlow

[<img src="cover.png" />](https://www.coursera.org/learn/natural-language-processing-tensorflow?)

En este curso, aprenderás:

- Construir sistemas de procesamiento del lenguaje natural utilizando TensorFlow

- Procesar texto, incluyendo la tokenización y la representación de frases como vectores

- Aplicar RNNs, GRUs y LSTMs en TensorFlow

- Entrene LSTMs en texto existente para crear poesía original y más

Si usted es un desarrollador de software que quiere construir algoritmos escalables impulsados por IA, necesita entender cómo utilizar 
las herramientas para construirlos. Esta Specializations le enseñará las mejores prácticas para utilizar TensorFlow, un popular 
marco de trabajo de código abierto para el aprendizaje automático. 

En el Curso 3 de la Specializations de deeplearning.ai TensorFlow, construirá sistemas de procesamiento de lenguaje natural utilizando TensorFlow. 
Aprenderá a procesar texto, incluyendo la tokenización y la representación de frases como vectores, para que puedan ser introducidos 
en una red neuronal. También aprenderá a aplicar RNNs, GRUs y LSTMs en TensorFlow. 

Por último, ¡llegará a entrenar una LSTM en un texto existente para crear poesía original! El curso de Aprendizaje Automático y 
la Especialización en Aprendizaje Profundo de Andrew Ng enseñan los principios más importantes y fundacionales del Aprendizaje 
Automático y el Aprendizaje Profundo. Esta nueva Especialización en TensorFlow de deeplearning.ai le enseña cómo utilizar 
TensorFlow para implementar esos principios de forma que pueda empezar a construir y aplicar modelos escalables a problemas del mundo real. 

Para desarrollar una comprensión más profunda de cómo funcionan las redes neuronales, le recomendamos que realice la Especialización en 
Aprendizaje Profundo.

## INDEX 0

- [Sentiment in text](#sentiment-in-text)
- [Weekly Assignment - Explore the BBC News Archive](#weekly-assignment---explore-the-bbc-news-archive)
- [Word Embeddings](#word-embeddings)
- [Weekly Assignment - More on the BBC News Archive](#weekly-assignment---more-on-the-bbc-news-archive)
- [Sequence Models](#sequence-models)
- [Weekly Assignment - Exploring overfitting in NLP](#weekly-assignment---exploring-overfitting-in-nlp)
- [Sequence models and literature](#sequence-models-and-literature)
- [Weekly Assignment - Generate Shakespeare-like text](#weekly-assignment---generate-shakespeare-like-text)

## Sentiment in text
[<- Return to INDEX 0](#index-0)

El primer paso para comprender el sentimiento en un texto, y en particular a la hora de entrenar una red neuronal para ello, es la tokenización de ese texto. Se trata del proceso de convertir el texto en valores numéricos, en el que un número representa una palabra o un carácter. Esta semana conocerá las API Tokenizer y pad_sequences de TensorFlow y cómo se pueden utilizar para preparar y codificar texto y frases con el fin de dejarlos listos para el entrenamiento de redes neuronales

### INDEX 1

- [A conversation with Andrew Ng 1](#a-conversation-with-andrew-ng-1)
- [Where to find the notebooks for this course](#where-to-find-the-notebooks-for-this-course)
- [Introduction](#introduction)
- [Word based encodings](#word-based-encodings)
- [Using APIs](#using-apis)
- [Have questions, issues or ideas?](#have-questions-issues-or-ideas)
- [Check out the code - Week 1 - (Lab 1)](#check-out-the-code---week-1---lab-1)
- [Notebook for lesson 1 - Week 1](#notebook-for-lesson-1---week-1)
- [Text to sequence](#text-to-sequence)
- [Looking more at the Tokenizer](#looking-more-at-the-tokenizer)
- [Padding](#padding)
- [Check out the code - Week 1 - (Lab 2)](#check-out-the-code---week-1---lab-2)
- [Notebook for lesson 2 - Week 1](#notebook-for-lesson-2---week-1)
- [Sarcasm, really?](#sarcasm-really)
- [Working with the Tokenizer](#working-with-the-tokenizer)
- [News Headlines dataset for sarcasm detection](#news-headlines-dataset-for-sarcasm-detection)
- [Check out the code - Week 1 - (Lab 3)](#check-out-the-code---week-1---lab-3)
- [Notebook for lesson 3 - Week 1](#notebook-for-lesson-3---week-1)
- [Week 1 Quiz](#week-1-quiz)
- [Week 1 Wrap up](#week-1-wrap-up)
- [Lecture Notes Week 1](#lecture-notes-week-1)

### A conversation with Andrew Ng 1
[<- Return to INDEX 1](#index-1)

### Where to find the notebooks for this course
[<- Return to INDEX 1](#index-1)

### Introduction
[<- Return to INDEX 1](#index-1)

### Word based encodings
[<- Return to INDEX 1](#index-1)

### Using APIs
[<- Return to INDEX 1](#index-1)

### Have questions, issues or ideas?
[<- Return to INDEX 1](#index-1)

### Check out the code - Week 1 - (Lab 1)
[<- Return to INDEX 1](#index-1)

### Notebook for lesson 1 - Week 1
[<- Return to INDEX 1](#index-1)

### Text to sequence
[<- Return to INDEX 1](#index-1)

### Looking more at the Tokenizer
[<- Return to INDEX 1](#index-1)

### Padding
[<- Return to INDEX 1](#index-1)

### Check out the code - Week 1 - (Lab 2)
[<- Return to INDEX 1](#index-1)

### Notebook for lesson 2 - Week 1
[<- Return to INDEX 1](#index-1)

### Sarcasm, really?
[<- Return to INDEX 1](#index-1)

### Working with the Tokenizer
[<- Return to INDEX 1](#index-1)

### News Headlines dataset for sarcasm detection
[<- Return to INDEX 1](#index-1)

### Check out the code - Week 1 - (Lab 3)
[<- Return to INDEX 1](#index-1)

### Notebook for lesson 3 - Week 1
[<- Return to INDEX 1](#index-1)

### Week 1 Quiz
[<- Return to INDEX 1](#index-1)

### Week 1 Wrap up
[<- Return to INDEX 1](#index-1)

### Lecture Notes Week 1
[<- Return to INDEX 1](#index-1)

## Weekly Assignment - Explore the BBC News Archive
[<- Return to INDEX 0](#index-0)

### Assignment Troubleshooting Tips

### Downloading your Notebook and refreshing your workspace

### Explore the BBC news archive

## Word Embeddings
[<- Return to INDEX 0](#index-0)

La semana pasada vio cómo utilizar el Tokenizer para preparar su texto para ser utilizado por una red neuronal convirtiendo las palabras en tokens numéricos, y secuenciando frases a partir de estos tokens. Esta semana aprenderá sobre Embeddings, donde estos tokens se mapean como vectores en un espacio de alta dimensión. Con Embeddings y ejemplos etiquetados, estos vectores pueden ajustarse para que las palabras con un significado similar tengan una dirección parecida en el espacio vectorial. Esto iniciará el proceso de entrenamiento de una red neuronal para comprender el sentimiento en el texto -- y comenzará observando críticas de películas, entrenando una red neuronal en textos etiquetados como "positivos" o "negativos" y determinando qué palabras de una frase impulsan esos significados.

### INDEX 2

- [A conversation with Andrew Ng 2](#a-conversation-with-andrew-ng-2)
- [Introduction Week 2](#introduction-week-2)
- [The IMDB dataset](#the-imdb-dataset)
- [IMDB reviews dataset](#imdb-reviews-dataset)
- [Looking into the details](#looking-into-the-details)
- [How can we use vectors?](#how-can-we-use-vectors)
- [More into the details](#more-into-the-details)
- [Checkout the code - week 2 - (Lab 1)](#checkout-the-code---week-2---lab-1)
- [Notebook for lesson 1 - week 2](#notebook-for-lesson-1---week-2)
- [Remember the sarcasm dataset?](#remember-the-sarcasm-dataset)
- [Building a classifier for the sarcasm dataset](#building-a-classifier-for-the-sarcasm-dataset)
- [Let's talk about the loss](#lets-talk-about-the-loss)
- [Checkout the code - week 2 - (Lab 2)](#checkout-the-code---week-2---lab-2)
- [Pre-tokenized datasets](#pre-tokenized-datasets)
- [Diving into the code (part 1) - week 2](#diving-into-the-code-part-1---week-2)
- [Subwords text encoder](#subwords-text-encoder)
- [Diving into the code (part 2) - week 2](#diving-into-the-code-part-2---week-2)
- [Checkout the code - week 2 - (Lab 3)](#checkout-the-code---week-2---lab-3)
- [Week 2 Quiz](#week-2-quiz)
- [Week 2 Wrap Up](#week-2-wrap-up)
- [Lecture Notes Week 2](#lecture-notes-week-2)

### A conversation with Andrew Ng 2
[<- Return to INDEX 2](#index-2)

### Introduction Week 2
[<- Return to INDEX 2](#index-2)

### The IMDB dataset
[<- Return to INDEX 2](#index-2)

### IMDB reviews dataset
[<- Return to INDEX 2](#index-2)

### Looking into the details
[<- Return to INDEX 2](#index-2)

### How can we use vectors?
[<- Return to INDEX 2](#index-2)

### More into the details
[<- Return to INDEX 2](#index-2)

### Checkout the code - week 2 - (Lab 1)
[<- Return to INDEX 2](#index-2)

### Notebook for lesson 1 - week 2
[<- Return to INDEX 2](#index-2)

### Remember the sarcasm dataset?
[<- Return to INDEX 2](#index-2)

### Building a classifier for the sarcasm dataset
[<- Return to INDEX 2](#index-2)

### Let's talk about the loss
[<- Return to INDEX 2](#index-2)

### Checkout the code - week 2 - (Lab 2)
[<- Return to INDEX 2](#index-2)

### Pre-tokenized datasets
[<- Return to INDEX 2](#index-2)

### Diving into the code (part 1) - week 2
[<- Return to INDEX 2](#index-2)

### Subwords text encoder
[<- Return to INDEX 2](#index-2)

### Diving into the code (part 2) - week 2
[<- Return to INDEX 2](#index-2)

### Checkout the code - week 2 - (Lab 3)
[<- Return to INDEX 2](#index-2)

### Week 2 Quiz
[<- Return to INDEX 2](#index-2)

### Week 2 Wrap Up
[<- Return to INDEX 2](#index-2)

### Lecture Notes Week 2
[<- Return to INDEX 2](#index-2)

## Weekly Assignment - More on the BBC News Archive
[<- Return to INDEX 0](#index-0)

### Diving deeper into the BBC News Archive

## Sequence Models
[<- Return to INDEX 0](#index-0)

En las dos últimas semanas, primero analizó la tokenización de palabras para obtener valores numéricos de ellas y, después, el uso de embeddings para agrupar palabras de significado similar en función de cómo estuvieran etiquetadas. Esto le proporcionó un buen análisis del sentimiento, aunque aproximado: palabras como "divertido" y "entretenido" pueden aparecer en una crítica de cine positiva, y "aburrido" y "soso", en una negativa. Pero el sentimiento también puede determinarse por la secuencia en la que aparecen las palabras. Por ejemplo, podría tener 'no divertido', que por supuesto es lo contrario de 'divertido'. Esta semana empezará a profundizar en diversos formatos de modelos que se utilizan en el entrenamiento de modelos para comprender el contexto en secuencia

### INDEX 3

- [A conversation with Andrew Ng 3](#a-conversation-with-andrew-ng-3)
- [Introduction week 3](#introduction-week-3)
- [Link to Andrew's sequence modeling course](#link-to-andrews-sequence-modeling-course)
- [LSTMs](#lstms)
- [More info on LSTMs](#more-info-on-lstms)
- [Implementing LSTMs in code](#implementing-lstms-in-code)
- [Check out the code - week 3 - (Lab 1 and Lab 2)](#check-out-the-code---week-3---lab-1-and-lab-2)
- [Accuracy and loss](#accuracy-and-loss)
- [A word from Laurence](#a-word-from-laurence)
- [Looking into the code](#looking-into-the-code)
- [Using a convolutional network](#using-a-convolutional-network)
- [Check out the code - week 3 - (Lab 3)](#check-out-the-code---week-3---lab-3)
- [Going back to the IMDB dataset](#going-back-to-the-imdb-dataset)
- [Check out the code - week 3 - (Lab 4)](#check-out-the-code---week-3---lab-4)
- [Tips from Laurence](#tips-from-laurence)
- [Exploring different sequence models (Lab 5 and Lab 6)](#exploring-different-sequence-models-lab-5-and-lab-6)
- [Week 3 Quiz](#week-3-quiz)
- [Week 3 Wrap up](#week-3-wrap-up)
- [Lecture Notes Week 3](#lecture-notes-week-3)

### A conversation with Andrew Ng 3
[<- Return to INDEX 3](#index-3)

### Introduction week 3
[<- Return to INDEX 3](#index-3)

### Link to Andrew's sequence modeling course
[<- Return to INDEX 3](#index-3)

### LSTMs
[<- Return to INDEX 3](#index-3)

### More info on LSTMs
[<- Return to INDEX 3](#index-3)

### Implementing LSTMs in code
[<- Return to INDEX 3](#index-3)

### Check out the code - week 3 - (Lab 1 and Lab 2)
[<- Return to INDEX 3](#index-3)

### Accuracy and loss
[<- Return to INDEX 3](#index-3)

### A word from Laurence
[<- Return to INDEX 3](#index-3)

### Looking into the code
[<- Return to INDEX 3](#index-3)

### Using a convolutional network
[<- Return to INDEX 3](#index-3)

### Check out the code - week 3 - (Lab 3)
[<- Return to INDEX 3](#index-3)

### Going back to the IMDB dataset
[<- Return to INDEX 3](#index-3)

### Check out the code - week 3 - (Lab 4)
[<- Return to INDEX 3](#index-3)

### Tips from Laurence
[<- Return to INDEX 3](#index-3)

### Exploring different sequence models (Lab 5 and Lab 6)
[<- Return to INDEX 3](#index-3)

### Week 3 Quiz
[<- Return to INDEX 3](#index-3)

### Week 3 Wrap up
[<- Return to INDEX 3](#index-3)

### Lecture Notes Week 3
[<- Return to INDEX 3](#index-3)

## Weekly Assignment - Exploring overfitting in NLP
[<- Return to INDEX 0](#index-0)

### Exploring overfitting in NLP

## Sequence models and literature
[<- Return to INDEX 0](#index-0)

Tomando todo lo que ha aprendido en el entrenamiento de una red neuronal basada en la PNL, pensamos que podría ser un poco divertido darle la vuelta a la tortilla y alejarse de la clasificación y utilizar sus conocimientos para la predicción. Dado un conjunto de palabras, podría concebir predecir la palabra con más probabilidades de seguir a una palabra o frase determinada, y una vez que lo haya hecho, volver a hacerlo una y otra vez. Con eso en mente, esta semana construirá un generador de poesía. Está entrenado con las letras de las canciones tradicionales irlandesas, ¡y puede utilizarse para producir por sí mismo versos de bella sonoridad!

### INDEX 4

- [A conversation with Andrew Ng 4](#a-conversation-with-andrew-ng-4)
- [Introduction Week 4](#introduction-week-4)
- [Looking into the code 1 week 4](#looking-into-the-code-1-week-4)
- [Preparing the training data](#preparing-the-training-data)
- [More on the training data](#more-on-the-training-data)
- [Finding what the next word should be](#finding-what-the-next-word-should-be)
- [Example](#example)
- [Predicting a word](#predicting-a-word)
- [Check out the code - week 4 - (Lab 1)](#check-out-the-code---week-4---lab-1)
- [Notebook for lesson 1 - week 4](#notebook-for-lesson-1---week-4)
- [Poetry](#poetry)
- [Link to the dataset](#link-to-the-dataset)
- [Looking into the code 2 week 4](#looking-into-the-code-2-week-4)
- [Laurence the poet](#laurence-the-poet)
- [Check out the code - week 4 - (Lab 2)](#check-out-the-code---week-4---lab-2)
- [Your next task](#your-next-task)
- [Link to generative text using a character-based RNN](#link-to-generative-text-using-a-character-based-rnn)
- [Week 4 quiz](#week-4-quiz)
- [Lecture Notes Week 4](#lecture-notes-week-4)
- [Reminder about the end of access to Lab Notebooks](#reminder-about-the-end-of-access-to-lab-notebooks)
- [Wrap up week 4](#wrap-up-week-4)
- [A conversation with Andre Ng Final](#a-conversation-with-andrew-ng-final)
- [Acknowledgments](#acknowledgments)

### A conversation with Andrew Ng 4
[<- Return to INDEX 4](#index-4)

### Introduction Week 4
[<- Return to INDEX 4](#index-4)

### Looking into the code 1 week 4
[<- Return to INDEX 4](#index-4)

### Preparing the training data
[<- Return to INDEX 4](#index-4)

### More on the training data
[<- Return to INDEX 4](#index-4)

### Finding what the next word should be
[<- Return to INDEX 4](#index-4)

### Example
[<- Return to INDEX 4](#index-4)

### Predicting a word
[<- Return to INDEX 4](#index-4)

### Check out the code - week 4 - (Lab 1)
[<- Return to INDEX 4](#index-4)

### Notebook for lesson 1 - week 4
[<- Return to INDEX 4](#index-4)

### Poetry
[<- Return to INDEX 4](#index-4)

### Link to the dataset
[<- Return to INDEX 4](#index-4)

### Looking into the code 2 week 4
[<- Return to INDEX 4](#index-4)

### Laurence the poet
[<- Return to INDEX 4](#index-4)

### Check out the code - week 4 - (Lab 2)
[<- Return to INDEX 4](#index-4)

### Your next task
[<- Return to INDEX 4](#index-4)

### Link to generative text using a character-based RNN
[<- Return to INDEX 4](#index-4)

### Week 4 quiz
[<- Return to INDEX 4](#index-4)

### Lecture Notes Week 4
[<- Return to INDEX 4](#index-4)

### Reminder about the end of access to Lab Notebooks
[<- Return to INDEX 4](#index-4)

### Wrap up week 4
[<- Return to INDEX 4](#index-4)

### A conversation with Andrew NG Final
[<- Return to INDEX 4](#index-4)

### Acknowledgments
[<- Return to INDEX 4](#index-4)

## Weekly Assignment - Generate Shakespeare-like text
[<- Return to INDEX 0](#index-0)

### Predicting the next word