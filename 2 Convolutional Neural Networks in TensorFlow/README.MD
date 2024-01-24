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

![img.png](ims%2FW1%2Fimg.png)

GPT
En el segundo curso de TensorFlow, los estudiantes avanzan significativamente en su aprendizaje. La primera semana se centra en aplicar conocimientos previos a un conjunto de datos más amplio, específicamente 25,000 imágenes de gatos y perros de Kaggle. Esto contrasta con el módulo anterior que trabajaba con aproximadamente 1000 imágenes de caballos y humanos. La intención es entrenar con un conjunto de datos más grande para abordar el desafío del sobreajuste, que es menos probable en conjuntos de datos amplios.

![img_1.png](ims%2FW1%2Fimg_1.png)

En la segunda semana, se introduce el aumento de datos como método para tratar el sobreajuste. TensorFlow ofrece herramientas para modificar imágenes (como girar, voltear, mover) durante el entrenamiento sin alterar el conjunto de datos original. Esto se hace en memoria para no editar las imágenes directamente.

Otra estrategia importante que se discute es el aprendizaje por transferencia. Los estudiantes aprenden a utilizar redes neuronales preentrenadas en grandes conjuntos de datos para mejorar el rendimiento en tareas específicas, aprovechando características ya aprendidas. Esto es fundamental para quienes no disponen de grandes volúmenes de datos.

![img_2.png](ims%2FW1%2Fimg_2.png)

Finalmente, en la cuarta semana, el curso aborda el aprendizaje con múltiples clases. Se mueve más allá de clasificaciones binarias, como gatos contra perros, a clasificaciones con tres o más categorías, como en el caso de piedra, papel o tijera.
El curso es descrito como integral y adecuado tanto para principiantes como para expertos.

### Where to find the notebooks for this course
[<- Return to INDEX 1](#index-1)

Todos los cuadernos de este curso pueden ejecutarse en Google Colab o en Coursera Labs. **No necesita tener configurado un entorno local para seguir los ejercicios de codificación.** Puede simplemente hacer clic en la insignia Open in Colab en la parte superior de los laboratorios no calificados, mientras que para las tareas, se le llevará automáticamente a Coursera Labs. 

Sin embargo, si desea ejecutarlos en su máquina local, los laboratorios no calificados y las asignaciones para cada semana se pueden encontrar en este 
[repositorio de Github](https://github.com/https-deeplearning-ai/tensorflow-1-public)
 bajo la carpeta **C2**. Si ya tiene git instalado en su ordenador, puede clonarlo con este comando:

```bash
git clone https://github.com/https-deeplearning-ai/tensorflow-1-public
```

Si no, por favor siga las guías 
[aquí](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
 para instalar git en su sistema operativo. Una vez que haya clonado el repositorio, puede hacer un **git pull** de vez en cuando para asegurarse de que recibe las últimas actualizaciones de los cuadernos.

Necesitará estos paquetes si va a ejecutar los cuadernos localmente:

```requirements
tensorflow==2.7.0
scikit-learn==1.0.1
pandas==1.1.5
matplotlib==3.2.2
seaborn==0.11.2
```

### A conversation with Andrew Ng 1
[<- Return to INDEX 1](#index-1)

![img_3.png](ims%2FW1%2Fimg_3.png)

El usuario se encuentra interesado en obtener un resumen de un contenido relacionado con la descarga de conjuntos de datos públicos de Internet, en particular de imágenes de gatos y perros, y cómo hacer que una red neuronal funcione con ellos. Se menciona que estos datos pueden ser confusos y contener sorpresas, como imágenes de personas con gatos o múltiples gatos en brazos. La solución propuesta implica el uso de TensorFlow y habilidades de programación en Python para filtrar y limpiar los datos. Se planea construir una red convolucional para detectar características específicas, como personas sosteniendo a los animales. 

![img_4.png](ims%2FW1%2Fimg_4.png)

Se destaca que se trabajará con un conjunto de datos claro de gatos y perros, pero se menciona la importancia de aprender habilidades que puedan aplicarse a conjuntos de datos menos limpios en el futuro. Además, se hace hincapié en que la realidad de la inteligencia artificial implica un trabajo significativo en la limpieza de datos y el uso de herramientas eficientes para ello. El objetivo final es entrenar una red neuronal para clasificar gatos y perros, y se anima al usuario a sumergirse en el contenido y practicar estas habilidades.

Para descargar un conjunto de datos públicos de Internet y hacer funcionar una red neuronal, especialmente en un contexto donde los datos pueden ser confusos o contener elementos inesperados, se requiere un enfoque estructurado y habilidades específicas. Este proceso incluye varios pasos clave:

1. **Adquisición de Datos:** Descargar el conjunto de datos de fuentes confiables. En el caso de imágenes de gatos y perros, se pueden encontrar numerosos conjuntos de datos públicos disponibles para fines de aprendizaje automático.

2. **Inspección y Limpieza de Datos:** Es crucial inspeccionar los datos para identificar y solucionar problemas comunes. Esto puede incluir archivos dañados o de longitud cero, imágenes irrelevantes o mal etiquetadas, y situaciones inusuales como personas sosteniendo a los animales. Se utilizan habilidades de programación, en este caso, con Python, para filtrar y limpiar el conjunto de datos.

3. **Preprocesamiento de Datos:** Adaptar las imágenes para que sean adecuadas para su uso en una red neuronal. Esto puede incluir redimensionar imágenes, normalizar valores de píxeles, y posiblemente aumentar los datos para mejorar la robustez del modelo.

4. **Construcción de la Red Neuronal:** Implementar una red neuronal convolucional (CNN) utilizando TensorFlow, una biblioteca popular para aprendizaje profundo. La CNN es adecuada para tareas de clasificación de imágenes debido a su capacidad para detectar características visuales complejas.

5. **Entrenamiento del Modelo:** Alimentar los datos limpios y preprocesados al modelo y ajustar los parámetros de la red neuronal. Este proceso implica definir la arquitectura de la red, seleccionar una función de pérdida, un optimizador, y entrenar el modelo con los datos de entrenamiento.

6. **Evaluación y Ajuste:** Evaluar el rendimiento del modelo utilizando un conjunto de datos de prueba. En función de los resultados, se pueden realizar ajustes en la arquitectura de la red, los parámetros de entrenamiento o el proceso de preprocesamiento.

7. **Aplicación Práctica:** Una vez entrenado y ajustado, el modelo se puede utilizar para clasificar nuevas imágenes de gatos y perros, o adaptarse a otros conjuntos de datos.

### The cats vs dogs dataset
[<- Return to INDEX 1](#index-1)

En el siguiente vídeo, verá el famoso 
[conjunto de datos de Kaggle Perros contra Gatos](https://www.kaggle.com/c/dogs-vs-cats)

![img_5.png](ims%2FW1%2Fimg_5.png)

Originalmente se trataba de un reto para construir un clasificador dirigido a los mejores profesionales del aprendizaje automático y la IA del mundo, pero la tecnología ha avanzado tan rápidamente que verá cómo puede hacerlo en tan sólo unos minutos con algo de programación sencilla de redes neuronales convolucionales.

También es un buen ejercicio para ver un conjunto de datos más grande, descargarlo y prepararlo para el entrenamiento, así como manejar algo de preprocesamiento de datos. Incluso datos como estos, que han sido cuidadosamente curados para usted, pueden tener errores -- ¡como notará con algunas imágenes corruptas!

Además, es posible que note algunas advertencias sobre datos EXIF faltantes o corruptos a medida que las imágenes se cargan en el modelo para el entrenamiento. No se preocupe por esto -- ¡no afectará a su modelo! :)


### Training with the cats vs. dogs dataset
[<- Return to INDEX 1](#index-1)

![img_6.png](ims%2FW1%2Fimg_6.png)

Hemos pasado del conjunto de datos de moda en el que las imágenes eran pequeñas y se centraban en el sujeto, a una nueva situación en la que teníamos imágenes de caballos y humanos y poses de acción. Usamos circunvoluciones para ayudarnos a identificar las características de la imagen, independientemente de su ubicación. Este es un buen manual para resolver algunos problemas comunes de ciencia de datos en lugares como Kaggle. 

![img_7.png](ims%2FW1%2Fimg_7.png)

A continuación analizaremos un antiguo concurso en el que os animaban a crear un clasificador para diferenciar gatos y perros. Si no estáis familiarizados con Kaggle, es donde suelen publicarse desafíos de aprendizaje automático con premios. Gatos contra perros fue uno de los temas más famosos de hace unos años. Las técnicas que acabas de aprender pueden aplicarse realmente a ese problema. Así que recapitulemos algunos de los conceptos. 

![img_8.png](ims%2FW1%2Fimg_8.png)

Una de las ventajas de TensorFlow y Keras es que si colocas tus imágenes en subdirectorios con nombre, una imagen generada las etiquetará automáticamente. Así que con el conjunto de datos de perros y gatos podrías hacerlo y ya tienes una enorme ventaja a la hora de crear el clasificador. 

Luego puedes subdividirlo en un conjunto de entrenamiento y un conjunto de validación. A continuación, puede utilizar los generadores de imágenes que se encuentran en esas carpetas. Para usar un generador de imágenes, debe crear una instancia de uno. 

![img_9.png](ims%2FW1%2Fimg_9.png)

 Si los datos aún no están normalizados, puede hacerlo con el parámetro de reescala. A continuación, llama al flujo desde el directorio para obtener un objeto generador. Para el conjunto de datos de entrenamiento, apuntará al directorio de entrenamiento y, a continuación, especificará el tamaño objetivo.

En este caso, las imágenes son de todas las formas y tamaños. Así que cambiaremos su tamaño a 150 por 150 sobre la marcha. Estableceremos el tamaño de los lotes en 20. Hay 2000 imágenes, por lo que utilizaremos 100 lotes de 20 cada uno. Como hay dos clases que queremos clasificar, siguen siendo un modo de clase binaria. 


![img_10.png](ims%2FW1%2Fimg_10.png)

Del mismo modo, para la validación, configuramos un generador y apuntamos al directorio de validación. 

![img_11.png](ims%2FW1%2Fimg_11.png)

Podemos explorar las convoluciones y las agrupaciones y el recorrido de la imagen a través de ellas. Es muy similar a lo que viste con 
los caballos y los humanos. Tiene tres series de convoluciones seguidas de una agrupación. Por supuesto, la imagen mide 150 por 150. 
Del mismo modo, hay una sola neurona con una activación sigmoidea en la salida. 

![img_12.png](ims%2FW1%2Fimg_12.png)

El resumen de las capas es muy similar al anterior, pero ten en cuenta que el tamaño cambia. Empezamos con 150 por 150. Así que la convolución reduce eso a 148 por 148. A partir de ahí, iremos hasta que terminemos con 17 x 17 que introduciremos en las capas densas. 

![img_13.png](ims%2FW1%2Fimg_13.png)

La compilación es como antes. Ahora recuerde que puede modificar la velocidad de aprendizaje ajustando el parámetro lr. 

![img_14.png](ims%2FW1%2Fimg_14.png)

Ahora, a entrenar, podemos llamar al generador de model.fit y pasarle el generador de entrenamiento y el generador de validación. Eso es todo. Como puedes ver, es muy similar a lo que construiste para caballos contra humanos. Así que veámoslo en acción.

### Looking at the notebook (Lab 1)
[<- Return to INDEX 1](#index-1)

Ahora que ya hemos hablado de lo que supone ampliar a datos del mundo real utilizando el conjunto de datos Gatos contra Perros, vamos a adentrarnos en un 
cuaderno [C2_W1_Lab_1_cats_vs_dogs.ipynb](notebooks%2FW1%2FC2_W1_Lab_1_cats_vs_dogs.ipynb)
 que muestra cómo realizar el reto por sí mismo. En el siguiente vídeo, verá un screencast de este cuaderno en acción. Entonces podrá probarlo usted mismo.

### Have questions, issues or ideas? Join our Community!
[<- Return to INDEX 1](#index-1)

¡Hola!

Hemos creado una comunidad para que usted pueda:

- Pedir ayuda sobre las tareas y otros contenidos del curso.

- Discutir temas del curso.

- Compartir sus conocimientos con otros alumnos.

- Crear su red de contactos.

- Enterarse de las novedades, eventos y concursos de DeepLearning.AI.

> Para acceder a la comunidad de este curso, marque la casilla que aparece a continuación para indicar que acepta utilizar la aplicación de forma responsable y, a continuación, haga clic en elbotón"Iniciar aplicación" .

Si es nuevo en la comunidad, haga clic en el botón "Iniciar aplicación"  para crear su cuenta y acceder a nuestra comunidad.

Hemos creado esta 
[Guía](https://community.deeplearning.ai/c/faq/391)
 del usuario 
[para usted](https://community.deeplearning.ai/c/faq/391)
. Asegúrese de consultar las directrices comunitarias 
[del Código de Conducta ](https://community.deeplearning.ai/c/faq/code-of-conduct/392)
. ¿Tiene problemas para acceder a nuestra comunidad después de pulsar el botón "Iniciar aplicación"? Rellene este
 [formulario](https://forms.gle/bQhd4kLS7xGBxz9U6)
 para explicar su problema y nos pondremos en contacto con usted.

¡Esperamos verle pronto en nuestra comunidad!

- El equipo de DeepLearning.AI

### Working through the notebook
[<- Return to INDEX 1](#index-1)

Lo primero que haremos es descargar el conjunto de imágenes. Se almacenan como un archivo zip que contiene 3.000 imágenes, 2.000 de las cuales usaremos para entrenamiento y 1.000 para pruebas. Una vez descargados, necesitará acceder al sistema operativo subyacente de la máquina virtual en la que se ejecuta este Colab. Esto está disponible en el espacio de nombres del sistema operativo. 

Este código descomprimirá los datos de perros y gatos que acabas de descargar en el directorio /tmp. Allí, los subdirectorios se volverán a crear, porque están almacenados que estaban en el archivo zip. 
```python
!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

import zipfile

# Unzip the archive
local_zip = './cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall()

zip_ref.close()
```
Lo siguiente es que establezcamos nuestros directorios como variables, para que podamos apuntar los generadores hacia ellos. Ahora, que tenemos los directorios como variables, podemos pasarlos al os.list aquí para tomar los archivos de esos directorios, y cargarlos en las listas de Python. 

```python
import os

base_dir = 'cats_and_dogs_filtered'

print("Contents of base directory:")
print(os.listdir(base_dir))

print("\nContents of train directory:")
print(os.listdir(f'{base_dir}/train'))

print("\nContents of validation directory:")
print(os.listdir(f'{base_dir}/validation'))
```
```commandline
Contents of base directory:
['train', 'validation', 'vectorize.py']

Contents of train directory:
['dogs', 'cats']

Contents of validation directory:
['dogs', 'cats']
```
Podemos ver que hay una lista de nombres de archivo como cat253.jpg, etc. En la pestaña de archivos, también podemos inspeccionar el sistema de archivos y en TMP, podemos ver nuestra carpeta gatos versus perros, dentro de la cual tenemos subdirectorios de entrenamiento y validación, que contienen gatos y perros directorios. Las imágenes, por supuesto, se almacenan allí. 

```python
import os

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])
```
```commandline
['cat.163.jpg', 'cat.409.jpg', 'cat.864.jpg', 'cat.166.jpg', 'cat.578.jpg', 'cat.250.jpg', 'cat.328.jpg', 'cat.733.jpg', 'cat.262.jpg', 'cat.818.jpg']
['dog.802.jpg', 'dog.498.jpg', 'dog.234.jpg', 'dog.68.jpg', 'dog.947.jpg', 'dog.451.jpg', 'dog.265.jpg', 'dog.134.jpg', 'dog.455.jpg', 'dog.881.jpg']
```
Si contamos las imágenes en cada directorio, podemos estar seguros de que tenemos las cantidades correctas de imágenes. 
Como podemos ver, tenemos mil de cada animal en Entrenamiento y 500 de cada uno en Pruebas para un total de 3000. 
```python
print('total training cat images :', len(os.listdir(      train_cats_dir ) ))
print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))

print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))
```
```commandline
total training cat images : 1000
total training dog images : 1000
total validation cat images : 500
total validation dog images : 500
```
A continuación, podemos visualizar algunos de los datos, para que podamos ver lo diverso que es. Este código establecería un matplotlib que es una biblioteca de Python para dibujar gráficos. Este código recogerá algunos gatos y perros al azar y los dibujará en una cuadrícula. Una vez dibujado, podemos ver que hay mucha diversidad en estas imágenes. 
```python
%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[ pic_index-8:pic_index]
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
```

![img_15.png](ims%2FW1%2Fimg_15.png)

Ahora, construyamos nuestra red neuronal. Importaremos TensorFlow, y luego definiremos nuestro modelo. Imprimiremos el resumen, y aquí puede ver la forma de salida de cómo la imagen pasó a través de las capas, y reducir gradualmente su tamaño a través de la convolución y agrupación. 
```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
```

```commandline
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 148, 148, 16)      448       
                                                                 
 max_pooling2d (MaxPooling2  (None, 74, 74, 16)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 72, 72, 32)        4640      
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 36, 36, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 34, 34, 64)        18496     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 17, 17, 64)        0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 18496)             0         
                                                                 
 dense (Dense)               (None, 512)               9470464   
                                                                 
 dense_1 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 9494561 (36.22 MB)
Trainable params: 9494561 (36.22 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

```
Aquí, es donde compilamos nuestro modelo, definiendo la función de pérdida y el optimizador. Aquí, es donde configuramos los dos generadores, señalándolos a los subdirectorios de capacitación y validación. Estos contienen subdirectorios propios, cada uno con gatos y perros. 
```python
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

```
Cuando lo ejecute, verás la copia impresa. Encontró 2.000 imágenes en dos clases, y ese es el entrenamiento, y 1.000 imágenes en dos clases, esa es la prueba.
```commandline
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
```
Ahora, podemos hacer el entrenamiento. Esta vez en lugar de usar model.fit, volveremos a usar model.fit generador porque nuestra fuente de datos son los generadores. Ahora, puedes ver que han pasado. 
```python
history = model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
            verbose=2
            )
```
Ahora, comenzaremos el entrenamiento y lo veremos progresar. Debería tomar de 3 a 3,5 minutos. Cuando está hecho, se puede ver que la precisión es de aproximadamente 73% y no está mal. No es genial, pero no está mal. 
```commandline
Epoch 1/15
100/100 - 13s - loss: 0.7102 - accuracy: 0.5400 - val_loss: 0.6728 - val_accuracy: 0.5450 - 13s/epoch - 130ms/step
Epoch 2/15
100/100 - 6s - loss: 0.6608 - accuracy: 0.6130 - val_loss: 0.6164 - val_accuracy: 0.6970 - 6s/epoch - 59ms/step
...
100/100 - 5s - loss: 0.0704 - accuracy: 0.9915 - val_loss: 1.3483 - val_accuracy: 0.7500 - 5s/epoch - 49ms/step
Epoch 15/15
100/100 - 6s - loss: 0.0339 - accuracy: 0.9920 - val_loss: 1.4264 - val_accuracy: 0.7400 - 6s/epoch - 62ms/step
```

### What you'll see next
[<- Return to INDEX 1](#index-1)

En el último vídeo, vio un screencast del cuaderno que muestra cómo construir un clasificador para Gatos contra Perros. Vio cómo, en algunos casos, no clasificaba correctamente a un gato, y le pedimos que intentara averiguar cómo podría solucionarlo. En el siguiente vídeo, verá una solución a esto.

### Fixing through cropping
[<- Return to INDEX 1](#index-1)

> Contexto: en el último video subio una foto de un gatito y el modelo NO lo reconoció como tal, entonces nos propuso hallar una forma de que el modelo
> lo clasificara correctamente.

![img_16.png](ims%2FW1%2Fimg_16.png)

¿Encontraste una solución? Bueno, por supuesto que la tuya puede ser diferente a la mía pero déjame mostrarte lo que hice en el caso del gato que mi modelo pensaba era un perro. Así que volvamos al cuaderno, y ejecutaremos el código. Subiré esta imagen para ver cómo se clasifica. Es una cosecha de gatos, y he aquí, se clasifica como un gato. 

![img_17.png](ims%2FW1%2Fimg_17.png)

Vamos a abrirlo, y compararlo con la imagen original, y veremos que sólo recortando pude conseguir que cambiara su clasificación. 
Debe haber habido algo en la imagen sin recortar que coincida con las características de un perro. Pensé que era un experimento muy interesante, ¿no? 
Ahora, ¿qué crees que el impacto de los cultivos podría haber tenido en el entrenamiento? 

¿Habría entrenado el modelo para demostrar que este era un gato mejor que una imagen sin recortar. Eso es algo para pensar, y algo para explorar en la siguiente lección, pero primero volvamos al libro de trabajo.

### Visualizing the effect of the convolutions
[<- Return to INDEX 1](#index-1)

![img_19.png](ims%2FW1%2Fimg_19.png)

Vale, en el video anterior echaste un vistazo a un cuaderno que entrenaba una red neuronal convolucional que clasificaba gatos contra perros. Ahora echemos un vistazo a cómo funcionó eso.

Volvamos al cuaderno y echemos un vistazo al código que grafica las salidas de las circunvoluciones en capas de tracción máxima. La clave para esto es entender la API model.layers, que le permite encontrar las salidas e iterar a través de ellas, creando un modelo de visualización para cada una.

Luego podemos cargar una imagen aleatoria en una matriz y pasarla al método de predicción del modelo de visualización.

La variable a seguir es display_grid que se puede construir a partir de x que se lee como un mapa de características y se procesa un poco para la visibilidad en el bucle central.


````python
import numpy as np
import random
from tensorflow.keras.utils import img_to_array, load_img

# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Prepare a random input image from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Scale by 1/255
x /= 255.0

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so you can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):

  if len(feature_map.shape) == 4:

    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))

    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
````
A continuación, renderizaremos cada una de las circunvoluciones de la imagen , además de su agrupación, luego otra convolución, etc.

![img_18.png](ims%2FW1%2Fimg_18.png)



### Looking at accuracy and loss
[<- Return to INDEX 1](#index-1)

Antes de irnos, echemos un vistazo rápido a trazar la historia de aprendizaje de este modelo. El objeto tiene valores de precisión y pérdida de entrenamiento , así como valores de exactitud de validación y pérdida de validación. 

Así que vamos a iterar sobre estos y trazar ellos. Ahora, si miras de cerca, no solo llamamos model.fit, dijimos que la historia es igual a model.fit. Así que ahora tenemos un objeto de historial que podemos consultar para los datos.
```python
history = model.fit(
            train_generator,
            epochs=15,
            validation_data=validation_generator,
            verbose=2
            )
```

Aquí puede ver que estoy usando el mismo objeto de historial, y estoy llamando a su propiedad de historial que pasa en ACC, lo que me da la precisión del modelo. Cuando lo ejecuto y grafico la precisión de entrenamiento y validación, podemos ver que mi entrenamiento fue hacia uno, mientras que mi validación se niveló en rango de 0.7 a 0.75. 

```python
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
```

![img_20.png](ims%2FW1%2Fimg_20.png)

Eso demuestra que mi modelo no es malo, pero realmente no gané nada después de dos épocas. Se ajusta muy bien a los datos de entrenamiento con los datos de validación necesarios.

![img_21.png](ims%2FW1%2Fimg_21.png)

Estos resultados se confirman en la pérdida donde podemos ver que después de dos épocas, mi pérdida de entrenamiento bajó muy bien, pero mi pérdida de validación subió. Así que, mi modelo es del 75 por ciento exacto después de dos épocas, y realmente no necesito entrenar más.


Recuerde también que acabamos de utilizar un subconjunto de los datos completos. El uso de todo el conjunto de datos probablemente arrojaría mejores resultados. Pero antes de hacerlo, veamos algunas otras opciones, y lo haremos en la siguiente lección.

### What have we seen so far?
[<- Return to INDEX 1](#index-1)

Al final del último vídeo, vio cómo explorar el historial de entrenamiento y descubrió un fenómeno interesante: Aunque la precisión del conjunto de datos de entrenamiento fue muy alta, vimos que después de sólo unas pocas épocas, el conjunto de validación se niveló. Esto es una clara señal de que estamos sobreajustando de nuevo. Utilizar más datos debería ayudar con esto, pero también hay otras técnicas que puede utilizar con conjuntos de datos más pequeños. ¡Y las veremos en la lección de la semana que viene!

### Week 1 Quiz
[<- Return to INDEX 1](#index-1)

1. What does flow_from_directory give you on the ImageDataGenerator?

   - [ ] The ability to easily load images for training
   - [ ] The ability to pick the size of training images
   - [ ] The ability to automatically label images based on their directory name
   - [X] All of the above

   > Correct
   > That's right! The flow_from_directory method takes the path to a directory & generates batches of augmented data.

2. If my Image is sized 150x150, and I pass a 3x3 Convolution over it, what size is the resulting image?

   - [ ] 450x450
   - [ ] 153x153
   - [ ] 150x150
   - [X] 148x148

   > Correct
   > Nailed it! Applying a 3x3 convolution would result in a 148x148 image.

3. If my data is sized 150x150, and I use Pooling of size 2x2, what size will the resulting image be?

   - [ ] 149x149
   - [ ] 300x300
   - [X] 75x75
   - [ ] 148x148

   > Correct
   > Nailed it! Applying 2x2 pooling would result in a 75x75 image.

4. If I want to view the history of my training, how can I access it?

   - [ ] Use a model.fit_generator
   - [X] Create a variable ‘history’ and assign it to the return of model.fit or model.fit_generator
   - [ ] Download the model and inspect it
   - [ ] Pass the parameter ‘history=true’ to the model.fit

   > Correct
   > Exactly! The History.history attribute is a record of training loss values and metrics values at successive epochs.

5. What’s the name of the API that allows you to inspect the impact of convolutions on the images?

   - [ ] The model.pools API
   - [ ] The model.convolutions API
   - [X] The model.layers API
   - [ ] The model.images API

   > Correct
   > The model.layers API allows you to access the individual layers of the model, where you can examine the convolutions applied to the images.

6. When exploring the graphs, the loss levelled out at about .75 after 2 epochs, but the accuracy climbed close to 1.0 after 15 epochs. What's the significance of this?

   - [ ] There was no point training after 2 epochs, as we overfit to the validation data
   - [X] There was no point training after 2 epochs, as we overfit to the training data
   - [ ] A bigger training set would give us better validation accuracy
   - [ ] A bigger validation set would give us better training accuracy

   > Correct
   > Those values indicate overfitting to the training data.

7. Why is the validation accuracy a better indicator of model performance than training accuracy?

   - [ ] It isn't, they're equally valuable
   - [ ] There's no relationship between them
   - [X] The validation accuracy is based on images that the model hasn't been trained with, and thus a better indicator of how the model will perform with new images.
   - [ ] The validation dataset is smaller, and thus less accurate at measuring accuracy, so its performance isn't as important

   > Correct
   > The validation set comprises data that the model has not seen during training, which helps evaluate the model’s ability to generalize to new, unseen data.

8. Why is overfitting more likely to occur on smaller datasets?

   - [ ] Because in a smaller dataset, your validation data is more likely to look like your training data
   - [ ] Because there isn't enough data to activate all the convolutions or neurons
   - [ ] Because with less data, the training will take place more quickly, and some features may be missed
   - [X] Because there's less likelihood of all possible features being encountered in the training process.

   > Correct
   > Undoubtedly! A smaller size decreases the likelihood that the model will recognize all possible features during training.

### Week 1 Wrap up
[<- Return to INDEX 1](#index-1)

![img_22.png](ims%2FW1%2Fimg_22.png)

Así que ya has llegado al final de la primera semana. Enhorabuena. Lo que me parece realmente inspirador de esta semana en particular es que lo 
que acabáis de hacer con los conjuntos de datos sobre gatos y perros como desafío científico de Kaggle, no hace mucho tiempo, 
solo estaba disponible para los mejores investigadores de las mejores universidades, desde arriba. 

Pero ahora, gracias a la explosión de la potencia de cálculo, a los conjuntos de datos abiertos como este y a cosas como TensorFlow, puedes hacerlo tú mismo, en el ordenador de tu casa. Eso me parece muy inspirador y muy emocionante. Así que gracias y prepárate para la segunda semana.

### Lecture Notes Week 1
[<- Return to INDEX 1](#index-1)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C2_W1.pdf](notes%2FC2_W1.pdf)
.

## Weekly Assignment - Attempt cats vs dogs kaggle challenge
[<- Return to INDEX 0](#index-0)

### Assignment Troubleshooting Tips

He aquí algunas directrices generales antes de entregar sus tareas en este curso. Téngalas en cuenta no sólo para la tarea de esta semana, sino también para las siguientes:

1. Por favor, no modifique ningún código fuera de las etiquetasSTART CODE HERE yEND CODE HERE. Su solución sólo debe colocarse entre estos marcadores para garantizar una calificación correcta. Modificar los parámetros de las funciones y otras celdas de prueba probablemente romperá el calificador. Si desea experimentar con ellas, puede hacerlo después de haber superado con éxito la tarea.

2. Si tiene más preguntas, cree un tema en la comunidad Discourse en lugar de en los foros de discusión de Coursera. Puede unirse 
[siguiendo las instrucciones aquí](https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/ungradedLti/XxU1e/have-questions-meet-us-on-discourse)
. Allí obtendrá ayuda más rápidamente porque varios mentores y sus compañeros de aprendizaje están supervisando los mensajes. Sólo asegúrese de crear el tema en la categoría correcta del curso.

### Programming Assignment: Cats vs Dogs

Esta semana, ha explorado una versión reducida del conjunto de datos Gatos contra Perros y la ha utilizado para entrenar una red neuronal convolucional. Vio que se sobreadaptaba muy rápidamente, a pesar de los grandes resultados obtenidos con el conjunto de entrenamiento. Una solución a la sobreadaptación es utilizar más datos tanto para el entrenamiento como para la validación, y ese es el ejercicio de esta semana: ¡construir un clasificador utilizando el conjunto de datos completo de Gatos contra Perros de 25.000 imágenes!

Tenga en cuenta de nuevo que, al cargar las imágenes, es posible que reciba advertencias sobre datos EXIF que faltan o están corruptos. No se preocupe por esto: faltan datos en las imágenes, pero no son datos visuales que vayan a afectar al entrenamiento.

Empecemos a construir un clasificador utilizando el conjunto de datos completo Gatos contra Perros de 25k imágenes.

Complete las tareas del cuaderno de 
[tareas](https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W1/assignment/C2W1_Assignment.ipynb)
 y suba aquí los archivos pertinentes para su calificación.

> La respuesta de esta tarea es: [C2W1_Assignment.ipynb](notebooks%2FW1%2FC2W1_Assignment.ipynb)

> Este cuaderno está alojado en Github, por lo que para guardar cualquier cambio deberá crear una copia del mismo dentro de su Drive. Puede hacerlo haciendo clic en la pestaña `Archivo` y luego en la opción `Guardar una copia en Drive`.

Deberá cargar dos archivos:

- El cuaderno (un archivo con extensión .ipynb ) que se utilizará para probar las siguientes funciones:

   - create_train_val_dirs
   
   - split_data
   
   - train_val_generators
   
   - create_model


. Duration: 2 minutos2 min
Tareas de programación: Gatos vs Perros
. Duration: 3 horas3 h
Semana 1
Gatos vs Perros
Tareas de programación: Gatos vs PerrosEstado: Traducido automáticamente
Traducido automáticamente
Este artículo tiene contenido sin traducir

Dado que este elemento tiene contenido que no está alojado en Coursera, no se traducirá al idioma que elegiste.
No has enviado tu trabajo. Debes obtener 80/100 puntos para aprobar.
Primera tarea de programación
Esta es tu primera tarea de programación correspondiente a curso.
Fecha Límite
Aprueba esta tarea antes del 28 de ene. 23:59 CST

Esta semana, ha explorado una versión reducida del conjunto de datos Gatos contra Perros y la ha utilizado para entrenar una red neuronal convolucional. Vio que se sobreadaptaba muy rápidamente, a pesar de los grandes resultados obtenidos con el conjunto de entrenamiento. Una solución a la sobreadaptación es utilizar más datos tanto para el entrenamiento como para la validación, y ese es el ejercicio de esta semana: ¡construir un clasificador utilizando el conjunto de datos completo de Gatos contra Perros de 25.000 imágenes!

Tenga en cuenta de nuevo que, al cargar las imágenes, es posible que reciba advertencias sobre datos EXIF que faltan o están corruptos. No se preocupe por esto: faltan datos en las imágenes, pero no son datos visuales que vayan a afectar al entrenamiento.

Empecemos a construir un clasificador utilizando el conjunto de datos completo Gatos contra Perros de 25k imágenes.


Complete las tareas del cuaderno de 
tareas
 y suba aquí los archivos pertinentes para su calificación.

Este cuaderno está alojado en Github, por lo que para guardar cualquier cambio deberá crear una copia del mismo dentro de su Drive. Puede hacerlo haciendo clic en la pestaña `Archivo` y luego en la opción `Guardar una copia en Drive`.

Deberá cargar dos archivos:

- El cuaderno (un archivo con extensión .ipynb ) que se utilizará para probar las siguientes funciones:

  * create_train_val_dirs

  * split_data

  * train_val_generators

  * create_model

- El historial de entrenamiento de su modelo (un archivo con extensión .p kl) que se utilizará para calificar la precisión de su modelo.

> IMPORTANTE PARA UNA CALIFICACIÓN CORRECTA:- No borre las celdas ya que incluyen metadatos importantes para la calificación- Rellene sus soluciones dentro de los espacios proporcionados. Puede añadir nuevas celdas pero éstas serán omitidas por el calificador.

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

![img.png](ims%2FW2%2Fimg.png)


En esta semana, se abordará el tema del aumento de imágenes y datos en el contexto del aprendizaje profundo. Esta técnica se utiliza ampliamente para ampliar los conjuntos de datos y mejorar el rendimiento de las redes neuronales. Se aprenderá a emplear las herramientas accesibles de TensorFlow para llevar a cabo este proceso de aumento de datos.

Como ejemplo, en la semana anterior se analizaron 25,000 imágenes de gatos y perros como un conjunto de datos de entrenamiento. Sin embargo, a veces, este tamaño de conjunto de datos no es suficiente para obtener resultados óptimos. Aumentar imágenes implica la creación de nuevos datos mediante transformaciones como rotaciones, distorsiones y otras técnicas. Esto permite abordar casos que quizás no estaban representados en el conjunto original, como gatos acostados en lugar de gatos siempre erguidos con orejas en una posición específica.

![img_1.png](ims%2FW2%2Fimg_1.png)

En TensorFlow, se enfatiza la importancia de no sobrecargar la memoria al aplicar aumentos. El proceso se realiza en la memoria a medida que las imágenes se cargan en la red neuronal para el entrenamiento. Esto significa que no se alteran los datos originales en el disco, sino que se generan temporalmente en la memoria.

La ventaja de este enfoque es que no se afectan los datos originales, lo que es esencial si se necesita experimentar con diferentes aumentos en el futuro. Además, es rápido y no aumenta significativamente los requisitos de memoria. En lugar de guardar todas las imágenes aumentadas, se procesan en la memoria y se transmiten a la red neuronal para su entrenamiento.

El aumento de datos e imágenes es una técnica fundamental en el aprendizaje profundo, y aunque la literatura académica sobre el tema puede ser limitada, es esencial para entrenar redes neuronales de manera efectiva. Por lo tanto, se recomienda consultar los materiales de esta semana para obtener más información sobre esta técnica importante.

### Image Augmentation
[<- Return to INDEX 2](#index-2)

Esta semana vamos a hablar mucho del Aumento de la imagen. 

El aumento de la imagen es una herramienta muy sencilla, pero muy poderosa, que le ayudará a evitar el sobreajuste de sus datos. El concepto es muy sencillo: si tiene datos limitados, las posibilidades de que disponga de datos que coincidan con posibles predicciones futuras también son limitadas y, lógicamente, cuantos menos datos tenga, menos posibilidades tendrá de obtener predicciones precisas para datos que su modelo aún no haya visto. En pocas palabras, si está entrenando un modelo para detectar gatos, y su modelo nunca ha visto el aspecto de un gato cuando está tumbado, es posible que no lo reconozca en el futuro.

El aumento simplemente modifica sus imágenes sobre la marcha mientras se entrena utilizando transformaciones como la rotación. Así, podría "simular" una imagen de un gato tumbado girando 90 grados un gato "de pie". De este modo, obtiene una forma barata de ampliar su conjunto de datos más allá de lo que ya tiene. 

![img_2.png](ims%2FW2%2Fimg_2.png)

Para saber más sobre la ampliación y las transformaciones disponibles, consulte 
https://keras.io/api/layers/preprocessing_layers/
 -- y tenga en cuenta que se denomina preprocesamientopor una razón muy poderosa: que no requiere que edite sus imágenes en bruto, ni las modifica por usted en el disco. Lo hace en memoria mientras está realizando el entrenamiento, lo que le permite experimentar sin afectar a su conjunto de datos. 

### Introducing augmentation
[<- Return to INDEX 2](#index-2)

![img_3.png](ims%2FW2%2Fimg_3.png)

Hasta este punto, hemos estado creando redes neuronales convolucionales que se entrenan para reconocer imágenes en clases binarias. Caballos o humanos, gatos o perros. Han funcionado bastante bien a pesar de tener cantidades relativamente pequeñas de datos con los que entrenar.

Sin embargo, corremos el riesgo de caer en la trampa del exceso de confianza provocado por el sobreajuste. Es decir, cuando el conjunto de datos es pequeño, tenemos relativamente pocos ejemplos y, como resultado, podemos tener algunos errores en nuestra clasificación. Probablemente nos hayas escuchado usar mucho el término sobreajuste y es importante entender qué es eso. 

![img_4.png](ims%2FW2%2Fimg_4.png)

Piensa que es muy bueno para detectar algo de un conjunto de datos limitado, pero te confundes cuando ves algo que no coincide con tus expectativas. Así, por ejemplo, imagina que estos son los únicos zapatos que has visto en tu vida. Luego, aprendes que estos son zapatos y este es el aspecto que tienen los zapatos. 

![img_5.png](ims%2FW2%2Fimg_5.png)

 Así que si te los mostrara, los reconocerías como zapatos, incluso si son de tamaños diferentes a los que cabría esperar. 

![img_6.png](ims%2FW2%2Fimg_6.png)

Pero si te mostrara esto, aunque es un zapato, probablemente no lo reconocerías como tal. En ese escenario, tienes un sobreajuste en tu comprensión de cómo es un zapato. 

![img_7.png](ims%2FW2%2Fimg_7.png)

No eras lo suficientemente flexible como para ver este tacón alto como un zapato porque todo tu entrenamiento y toda tu experiencia en el aspecto de los zapatos son estas botas de montaña.

Ahora bien, este es un problema habitual en los clasificadores de entrenamiento, sobre todo cuando se dispone de datos limitados. Si lo piensas bien, necesitarías un conjunto de datos infinito para crear un clasificador perfecto, pero podría llevar demasiado tiempo entrenarlo.

Por eso, en esta lección, quiero analizar algunas herramientas que tienes a tu disposición para hacer que tus conjuntos de datos más pequeños sean más efectivos. Empezaremos con un concepto simple, el aumento. 

![img_8.png](ims%2FW2%2Fimg_8.png)

Al usar redes neuronales convolucionales, pasamos convoluciones sobre una imagen para aprender características particulares. 
Tal vez sean las orejas puntiagudas de un gato, dos patas en vez de cuatro para un humano, ese tipo de cosas. Las convoluciones 
han sido muy buenas para detectarlas si son claras y diferenciadas en la imagen. 

![img_9.png](ims%2FW2%2Fimg_9.png)

Pero si pudiéramos ir más allá, ¿qué pasaría si, por ejemplo, pudiéramos transformar la imagen del gato para que coincidiera con otras imágenes de gatos en las que las orejas están orientadas de manera diferente? Así pues, si la red nunca ha sido entrenada para ver una imagen de un gato recostado de esta forma, es posible que no la reconozca. 

Si no tienes los datos de un gato reclinado, podrías acabar en una situación de sobreajuste. Sin embargo, si incorporas las imágenes al entrenamiento con medidas de aumento, como una rotación, es posible que se detecte la característica. Incluso si no tienes un gato reclinado, tu gato erguido, al girar, podría acabar teniendo el mismo aspecto.

### Start Coding...
[<- Return to INDEX 2](#index-2)

Bien, ahora que hemos visto la implementación del Aumento de Imagen en Keras, profundicemos en el código.

![img_10.png](ims%2FW2%2Fimg_10.png)

Puede ver más sobre las diferentes APIs en el sitio de Keras aquí: 
https://keras.io/preprocessing/image/

### Coding augmentation with ImageDataGenerator
[<- Return to INDEX 2](#index-2)

![img_11.png](ims%2FW2%2Fimg_11.png)

Entonces, si recuerdas la clase generadora de imágenes que usamos anteriormente, en realidad tiene la capacidad de hacerlo por ti. De hecho, ya has aumentado un poco la imagen con ella cuando cambias la escala al cargarla. Eso te ahorró tener que convertir todas tus imágenes en el sistema de archivos y luego cargarlas. Simplemente cambiaste la escala sobre la marcha.

Así que echemos un vistazo a algunas de las otras opciones. A continuación te explicamos cómo puedes usar un montón de opciones de aumento de imagen, añadiendo el generador de imágenes a la posibilidad de cambiar la escala.

![img_12.png](ims%2FW2%2Fimg_12.png)

El rango de rotación es un rango de 0-180 grados con el que se pueden rotar imágenes de forma aleatoria. Por lo tanto, en este caso, la imagen rotará aleatoriamente entre 0 y 40 grados. 

Al desplazarse, la imagen se mueve dentro de su marco. Muchas imágenes tienen el sujeto centrado. Así que si entrenamos basándonos en ese tipo de imágenes, podríamos sobreadaptarnos a ese escenario. Estos parámetros especifican, como proporción del tamaño de la imagen, cuánto debemos mover aleatoriamente al sujeto. Así que, en este caso, podríamos compensarlo en un 20 por ciento vertical u horizontalmente.

La cizalladura también es bastante poderosa. Así, por ejemplo, consideremos la imagen de la derecha. Sabemos que es una persona. Pero en nuestro set de entrenamiento, no tenemos ninguna imagen de una persona con esa orientación. Sin embargo, tenemos una imagen como esta, en la que la persona tiene una orientación similar. Así que si cortamos a esa persona inclinándola a lo largo del eje x, terminaremos en una postura similar. Eso es lo que nos da el parámetro shear_range. Cortará la imagen en cantidades aleatorias hasta alcanzar la parte especificada de la imagen. 
 Por lo tanto, en este caso, cortará hasta un 20 por ciento de la imagen. 

![img_13.png](ims%2FW2%2Fimg_13.png)

El zoom también puede ser muy eficaz. Por ejemplo, considere la imagen de la derecha. Obviamente es una mujer que mira hacia la derecha. Nuestra imagen de la izquierda es del set de entrenamiento de humanos o 
caballos. Es muy similar, pero se ha reducido para ver a la persona completa. Si ampliamos la imagen de entrenamiento, podríamos 
terminar con una imagen muy similar a la de la derecha. 

Por lo tanto, si ampliamos la imagen mientras entrenamos, podríamos ver ejemplos más generalizados como este. Así que ampliamos con un código como este. El 0.2 es una parte relativa de la imagen en la que ampliarás. Por lo tanto, en este caso, los acercamientos serán aleatorios de hasta un 20 por ciento del tamaño de la imagen. 

![img_15.png](ims%2FW2%2Fimg_15.png)

Otra herramienta útil es el volteo horizontal. Por ejemplo, si consideras la imagen de la derecha, es posible que no podamos clasificarla correctamente, ya que nuestros datos de entrenamiento no contienen la imagen de una mujer con la mano izquierda levantada, sino la imagen de la izquierda, donde el brazo derecho del sujeto está levantado. 

 Por lo tanto, si la imagen se volteara horizontalmente, se asemejaría más estructuralmente a la imagen de la derecha y no podríamos ajustarla demasiado a los elevadores del brazo derecho. Para activar el volteo horizontal aleatorio, solo tienes que decir que horizontal_flip es igual a true y las imágenes se voltearán de forma aleatoria.

![img_16.png](ims%2FW2%2Fimg_16.png)

Por último, solo especificamos el modo de relleno. Esto rellena cualquier píxel que se haya podido perder a causa de las operaciones. Me quedaré con el más cercano, que usa los vecinos de ese píxel para tratar de mantener la uniformidad. Consulte la documentación de los cursores para ver otras opciones. Así que ese es el concepto de aumento de imágenes. Analicemos ahora el contraste entre gatos y perros entrenados con y sin aumento, para que podamos ver el impacto que esto tiene.

### Looking at the notebook (Lab 1)
[<- Return to INDEX 2](#index-2)

Ahora que ha aprendido algunos de los fundamentos del aumento de datos, veámoslo en acción en el clasificador Gatos contra Perros.

En primer lugar, ejecutaremos Gatos contra Perros sin aumento, y exploraremos lo rápido que sobreajusta.

Si desea ejecutar el cuaderno usted mismo, puede encontrarlo 
aquí [C2_W2_Lab_1_cats_v_dogs_augmentation.ipynb](notebooks%2FW2%2FC2_W2_Lab_1_cats_v_dogs_augmentation.ipynb)
.

### Demonstrating overfitting in cats vs dogs
[<- Return to INDEX 2](#index-2)

![img_17.png](ims%2FW2%2Fimg_17.png)

Echemos un vistazo a la formación de gatos frente a perros con un conjunto de datos más pequeño durante un ciclo más largo. Voy a empezar el entrenamiento y veremos que recogió las 2.000 imágenes de entrenamiento, pero ten en cuenta que estamos entrenando para 100 épocas. 

![img_19.png](ims%2FW2%2Fimg_19.png)

Después de la primera época, podemos ver que nuestra precisión es 0.5345 y una precisión de validación es 0.5290. Vigila esas cifras. Veámoslos por algunas épocas más. Observaremos las cifras de precisión y validación. Después de ocho épocas, la precisión se acerca a 0.8, pero la precisión de validación ha ralentizado su crecimiento. Así que ahora saltemos adelante hasta el final.

![img_18.png](ims%2FW2%2Fimg_18.png)

Voy a trazar la precisión y la pérdida total de 100 épocas. Podemos ver en esta cifra que el entrenamiento alcanza casi un 100 por ciento de precisión en poco más de 20 épocas. Mientras tanto, la validación superó en alrededor del 70 por ciento, y eso es sobreajuste claramente se demostró. En otras palabras, la red neuronal fue excelente al encontrar una correlación entre las imágenes y las etiquetas de los gatos frente a los perros para las 2.000 imágenes en las que fue entrenado, pero una vez que trató de predecir las imágenes que antes no había visto, era aproximadamente un 70 por ciento de precisión.

Es un poco como el ejemplo de los zapatos de los que hablamos antes. Así que en el siguiente video, vamos a echar un vistazo al impacto de agregar aumento a esto.

### The impact of augmentation on Cats vs. Dogs
[<- Return to INDEX 2](#index-2)

Ahora que hemos visto cómo sobreajusta, veamos a continuación cómo, con una simple modificación de código, podemos añadir aumento de datos a la misma Red Neuronal Convolucional para ver cómo nos proporciona mejores datos de entrenamiento que sobreajustan menos. El cuaderno utilizado en el siguiente vídeo seguirá siendo el que 
acaba de utilizar en el screencast anterior [C2_W2_Lab_1_cats_v_dogs_augmentation.ipynb](notebooks%2FW2%2FC2_W2_Lab_1_cats_v_dogs_augmentation.ipynb)
. Por favor, vuelva a abrirlo si quiere seguir el vídeo.

### Adding augmentation to cats vs dogs
[<- Return to INDEX 2](#index-2)

![img_20.png](ims%2FW2%2Fimg_20.png)

En el video anterior, analizamos el entrenamiento de un pequeño conjunto de datos de gatos frente a perros, y vimos cómo el sobreajuste se produjo relativamente temprano en el entrenamiento, lo que nos lleva a una falsa sensación de seguridad acerca de lo bien que la red neuronal podría funcionar. Echemos ahora un vistazo al impacto de agregar aumento de imagen a la formación. 

![img_21.png](ims%2FW2%2Fimg_21.png)

Aquí tenemos exactamente el mismo código, excepto que le hemos agregado el código de aumento de la imagen. Empezaré el entrenamiento, y veremos que tenemos 2.000 imágenes de entrenamiento en dos clases.

![img_22.png](ims%2FW2%2Fimg_22.png)

 Al comenzar el entrenamiento, veremos inicialmente que la precisión es menor que con la versión no aumentada que hicimos anteriormente. Esto se debe a los efectos aleatorios del procesamiento de imágenes diferente que se está haciendo. A medida que se ejecuta durante algunas épocas más, verá la precisión subiendo lentamente. 

![img_23.png](ims%2FW2%2Fimg_23.png)

Saltaré hacia adelante para ver las últimas épocas, y para cuando lleguemos a la última, nuestro modelo tiene un 86 por ciento de precisión en los datos de entrenamiento, y un 81 por ciento en los datos de prueba. Así que vamos a trazar esto. 

![img_24.png](ims%2FW2%2Fimg_24.png)

Podemos ver que la precisión de entrenamiento y validación, y la pérdida están en realidad en el paso entre sí. Esta es una clara señal de que hemos resuelto el sobreajuste que teníamos antes. 

Si bien nuestra precisión es un poco menor, también está tendencia hacia arriba, así que quizás más épocas nos acerquen al 100 por ciento. ¿ Por qué no lo intentas?

### Image Augmentation with Horses vs Humans! (Lab 2)
[<- Return to INDEX 2](#index-2)

Habiendo visto claramente el impacto que el aumento da a Gatos contra Perros, volvamos ahora al conjunto de datos Caballos contra Humanos del curso 1, y echemos un vistazo para ver si los algoritmos de aumento ayudan en este caso. Aquí tiene el 
cuaderno [C2_W2_Lab_2_horses_v_humans_augmentation.ipynb](notebooks%2FW2%2FC2_W2_Lab_2_horses_v_humans_augmentation.ipynb)
 por si quiere seguir el siguiente vídeo.

### Exploring augmentation with horses vs humans
[<- Return to INDEX 2](#index-2)

![img_25.png](ims%2FW2%2Fimg_25.png)

Por supuesto, el aumento de imágenes no es la bala mágica para curar el sobreajuste. Realmente ayuda tener una gran diversidad de imágenes. Así, por ejemplo, si miramos el conjunto de datos de caballos o humanos y lo entrenamos para las mismas épocas, entonces podemos echar un vistazo a su comportamiento. 

![img_26.png](ims%2FW2%2Fimg_26.png)

 Así que voy a empezar a entrenar y mostrar las 100 épocas. Lo aceleré un poco para ahorrar tiempo. Mientras observas, verás que la precisión de la prueba sube constantemente. Al principio, la precisión de validación parece estar en paso, pero luego la verá variando salvajemente. Lo que sucede aquí es que, a pesar del aumento de la imagen, la diversidad de imágenes sigue siendo demasiado escasa y el conjunto de validación también puede estar mal diseñado, es decir, que el tipo de imagen que contiene está demasiado cerca de las imágenes del conjunto de entrenamiento

![img_27.png](ims%2FW2%2Fimg_27.png)

Si esperas los datos para ti mismo, verás que ese es el caso. Por ejemplo, los humanos casi siempre están de pie y en el centro de la imagen, tanto en los conjuntos de entrenamiento como de validación, por lo que aumentar la imagen cambiará para que parezca algo que no se parezca a lo que hay en el conjunto de validación. Así que para cuando el entrenamiento se haya completado, podemos ver el mismo patrón. La precisión del entrenamiento tiende hacia el 100%, pero la validación está fluctuando en los años 60 y 70.

![img_28.png](ims%2FW2%2Fimg_28.png)

Vamos a trazar esto, podemos ver que la precisión del entrenamiento sube constantemente de la manera que quisiéramos, pero la validación fluctuó como una locura. Entonces, lo que podemos aprender de esto es que el aumento de la imagen introduce un elemento aleatorio a las imágenes de entrenamiento, pero si el conjunto de validación no tiene la misma aleatoriedad, entonces sus resultados pueden fluctuar así.

Así que ten en cuenta que no solo necesitas un amplio conjunto de imágenes para entrenar, sino que también las necesitas para probar o el aumento de imágenes no te ayudará mucho.

### What have we seen so far? 2
[<- Return to INDEX 2](#index-2)

Esta semana, ha examinado la herramienta realmente útil que le ofrece TensorFlow con el aumento de imágenes. Con ella, puede simular eficazmente un conjunto de datos más grande a partir de uno más pequeño con herramientas para mover las imágenes alrededor del marco, sesgarlas, rotarlas y mucho más. Esta puede ser una herramienta eficaz para corregir el sobreajuste.

### Week 2 Quiz
[<- Return to INDEX 2](#index-2)

1. How do you use Image Augmentation in TensorFlow?

   - [X] Using parameters to the ImageDataGenerator
   - [ ] You have to write a plugin to extend tf.layers
   - [ ] With the keras.augment API
   - [ ] With the tf.augment API
   
   > Correct
   > TensorFlow's ImageDataGenerator allows you to specify augmentation techniques directly as parameters, making it easier to augment images on-the-fly during training.

2. If my training data only has people facing left, but I want to classify people facing right, how would I avoid overfitting?

   - [ ] Use the ‘flip_vertical’ parameter around the Y axis
   - [X] Use the ‘horizontal_flip’ parameter
   - [ ] Use the ‘flip’ parameter and set ‘horizontal’
   - [ ] Use the ‘flip’ parameter

   > Correct
   > The 'horizontal_flip' parameter in the ImageDataGenerator generates images with either horizontal orientation, thereby increasing the diversity of the training data and helping prevent overfitting.

3. After adding data augmentation and using the same batch size and steps per epoch, you noticed that each training epoch became a little slower than when you trained without it. Why?

   - [ ] Because there is more data to train on
   - [X] Because the image preprocessing takes cycles
   - [ ] Because the training is making more mistakes
   - [ ] Because the augmented data is bigger

   > Correct
   > Image augmentation requires additional computational resources for image preprocessing, which can make each epoch take longer to complete.

4. What does the fill_mode parameter do?

   - [ ] There is no fill_mode parameter
   - [ ] It creates random noise in the image
   - [X] It attempts to recreate lost information after a transformation like a shear
   - [ ] It masks the background of an image

   > Correct
   > The fill_mode parameter specifies how to fill in newly created pixels which can appear after a rotation or a width/height shift.

5. When using Image Augmentation with the ImageDataGenerator, what happens to your raw image data on-disk?

   - [ ] It gets overwritten, so be sure to make a backup
   - [ ] A copy is made and the augmentation is done on the copy
   - [X] Nothing, all augmentation is done in-memory
   - [ ] It gets deleted

   > Correct
   > ImageDataGenerator performs augmentations in-memory without altering the original images on disk, ensuring your data remains unchanged.

6. How does Image Augmentation help solve overfitting?

   - [ ] It slows down the training process
   - [X] It manipulates the training set to generate more scenarios for features in the images
   - [ ] It manipulates the validation set to generate more scenarios for features in the images
   - [ ] It automatically fits features to images by finding them through image processing techniques

   > Correct
   > By artificially creating variations of the training images, image augmentation exposes the model to a broader range of scenarios, which can help the model generalize better, thereby reducing overfitting.

7. When using Image Augmentation my training gets...

   - [X] Slower
   - [ ] Faster
   - [ ] Stays the Same
   - [ ] Much Faster

   > Correct
   > Although beneficial, image augmentation introduces additional computations for altering images during training, which can slow down the overall training process.

8. Using Image Augmentation effectively simulates having a larger data set for training.

   - [ ] False
   - [X] True

   > Correct
   > Image augmentation artificially inflates the dataset by generating many plausible variations of each image, providing the benefits of a larger dataset without the need for more actual data.

### Week 2 Wrap up
[<- Return to INDEX 2](#index-2)

![img_29.png](ims%2FW2%2Fimg_29.png)

Así que, felicidades por terminar la segunda semana. Espero que hayáis disfrutado esta semana tanto como yo, porque el aumento de imágenes es, francamente, una de mis cosas favoritas para jugar en TensorFlow.

Me gusta mucho el hecho de que cuando tienes un conjunto de datos pequeño, como este con caballos contra humanos, puedas manipular ese conjunto de datos sin cambiar las imágenes subyacentes para abrir escenarios completamente nuevos para el entrenamiento y poder entrenar como vimos, como un hombre acostado o una mujer mirando a la izquierda, y poder entrenar para eso incluso si no tienes esas imágenes en tu set de entrenamiento. Pero la semana que viene, vamos a hacer algo realmente emocionante y se llama aprendizaje por transferencia. Así que estad atentos.

### Lecture Notes Week 2
[<- Return to INDEX 2](#index-2)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C2_W2.pdf](notes%2FC2_W2.pdf)
.

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

![img.png](ims%2FW3%2Fimg.png)

En la clase, se discutió el concepto de aprendizaje por transferencia en el contexto del aprendizaje profundo y cómo TensorFlow facilita su implementación con unas pocas líneas de código. En lugar de entrenar una red neuronal desde cero, se explicó cómo se pueden descargar modelos de código abierto previamente entrenados en grandes conjuntos de datos y utilizarlos como punto de partida para tareas específicas. Este proceso se conoce como aprendizaje por transferencia.

![img_1.png](ims%2FW3%2Fimg_1.png)

Se mencionó que al descargar modelos entrenados previamente, se aprovecha el conocimiento adquirido de una amplia variedad de imágenes y características que pueden no estar disponibles en el conjunto de datos propio. Esto permite acelerar el proceso de entrenamiento y mejorar la eficiencia de las redes neuronales convolucionales. Además, se destacó la importancia de la comunidad de IA y cómo la compartición de conocimientos y código abierto ha impulsado el avance rápido de la inteligencia artificial.

![img_2.png](ims%2FW3%2Fimg_2.png)

El proceso de implementación del aprendizaje por transferencia en TensorFlow se describió como relativamente sencillo, donde se pueden descargar modelos, configurarlos y ajustarlos según las necesidades específicas. También se mencionó que los estudiantes aprenderán a agregar una red neuronal profunda a modelos preentrenados y a entrenarlos para lograr una mayor eficiencia y precisión en la clasificación de imágenes. En resumen, el aprendizaje por transferencia se presentó como una técnica valiosa para acelerar y mejorar el rendimiento de los modelos de aprendizaje profundo.

### Understanding transfer learning: the concepts
[<- Return to INDEX 3](#index-3)

![img_3.png](ims%2FW3%2Fimg_3.png)

En las lecciones anteriores, analizaste la creación de un clasificador binario que predijera gatos contra perros o caballos contra humanos. También viste cómo puede ocurrir un sobreajuste y exploraste algunas prácticas para evitarlo. El problema con ellas, por supuesto, es que los datos de entrenamiento eran muy pequeños y hay un número limitado de funciones comunes que se pueden extraer, incluso si utilizamos algunos trucos, como el aumento de imágenes.

Pero en ambos casos, construiste el modelo desde cero. ¿Y si pudieras tomar un modelo existente que se basa en muchos más datos y utilizar las funciones que ese modelo ha aprendido? Ese es el concepto de aprendizaje por transferencia, y lo exploraremos en esta lección.

![img_4.png](ims%2FW3%2Fimg_4.png)

 Así, por ejemplo, si visualizas tu modelo de esta manera con una serie de capas convolucionales antes de que la capa densa ocupe el lugar de salida, 

![img_5.png](ims%2FW3%2Fimg_5.png)

introduzcas tus datos en la capa superior, la red aprenderá las circunvoluciones que identifican las entidades de tus datos y todo eso. 

![img_6.png](ims%2FW3%2Fimg_6.png)

Pero consideremos el modelo de otra persona, quizás uno mucho más sofisticado que el suyo, basado en muchos más datos. Tienen capas convolucionales y aquí están intactas con características que ya se han aprendido.

![img_7.png](ims%2FW3%2Fimg_7.png)


De este modo, puede bloquearlas en lugar de volver a capacitarlas en sus datos, y hacer que simplemente extraigan las características de sus datos utilizando las convoluciones que ya han aprendido.

![img_8.png](ims%2FW3%2Fimg_8.png)

A continuación, puede tomar un modelo que se haya entrenado con conjuntos de datos muy grandes y utilizar las convoluciones que aprendió al clasificar sus datos. Si recuerda cómo se crean las convoluciones y cómo se utilizan para identificar entidades concretas, así

como el recorrido de una entidad por la red, tiene sentido utilizarlas y, a continuación, volver a entrenar las capas densas de ese modelo con sus datos. Por supuesto, bueno, lo habitual es bloquear todas las circunvoluciones. No tienes que hacerlo.


También puedes optar por volver a entrenar algunas de las inferiores, ya que pueden estar demasiado especializadas para las imágenes que tienes a mano. Se necesita un poco de ensayo y error para descubrir la combinación correcta.

![img_9.png](ims%2FW3%2Fimg_9.png)

Tomemos, pues, un modelo de última generación bien entrenado. Hay uno llamado Inception, del que puedes obtener más información en su sitio. 

![img_10.png](ims%2FW3%2Fimg_10.png)

Esto ha sido entrenado previamente con un conjunto de datos de ImageNet, que contiene 1,4 millones de imágenes en 1000 clases diferentes.

### Coding transfer learning from the inception model
[<- Return to INDEX 3](#index-3)

![img_11.png](ims%2FW3%2Fimg_11.png)

Así que echemos un vistazo a cómo lograríamos esto en el código. Empezaremos con las entradas. En concreto, utilizaremos la API de capas de keras para seleccionar las capas y entender cuáles queremos usar y cuáles queremos volver a entrenar

![img_12.png](ims%2FW3%2Fimg_12.png)

En esta URL se guarda una copia de los pesos previamente entrenados para la red neuronal inicial. Piense en ello como una instantánea del modelo después de haberlo entrenado.
Son los parámetros los que luego se pueden cargar en el esqueleto del modelo para convertirlo de nuevo en un modelo entrenado. 

![img_13.png](ims%2FW3%2Fimg_13.png)

Ahora, si queremos usar Inception, es una suerte que keras tenga la definición del modelo integrada. Así que crea una instancia con la forma de entrada que desee para sus datos y especifica que no quiere utilizar los pesos incorporados, sino la instantánea que acaba de descargar.

 El inception V3 tiene una capa totalmente conectada en la parte superior. Por lo tanto, al establecer include_top en false, estás especificando que quieres ignorarlo y pasar directamente a las convoluciones.

![img_14.png](ims%2FW3%2Fimg_14.png)

Ahora que he creado una instancia de mi modelo previamente entrenado, puedo recorrer en iteración sus capas y bloquearlas, diciendo que no se podrán entrenar con este código. 

![img_15.png](ims%2FW3%2Fimg_15.png)

Luego puedes imprimir un resumen de tu modelo previamente entrenado con este código, pero prepárate, es enorme. 

![img_16.png](ims%2FW3%2Fimg_16.png)

No hay forma de que pueda caber todo en una diapositiva, incluso si utilizo una fuente de dos puntos como esta. Esto es probablemente menos del 10 por ciento del resumen del modelo. Pruébelo usted mismo en el cuaderno.

### Adding your DNN
[<- Return to INDEX 3](#index-3)

En el vídeo anterior vio cómo tomar las capas de un modelo existente, y hacerlas de modo que no se reentrenen -- es decir, congela (o bloquea) las convoluciones ya aprendidas en su modelo. Ahora, tendrá que añadir su propia DNN en la parte inferior de éstas, que podrá reentrenar a sus datos. En el siguiente vídeo, verá cómo hacerlo...

### Coding your own model with transferred features
[<- Return to INDEX 3](#index-3)

![img_17.png](ims%2FW3%2Fimg_17.png)

Todas las capas tienen nombres, por lo que puede buscar el nombre de la última capa que desea utilizar. Si inspecciona el resumen, verá que las capas inferiores se han convertido en 3 por 3. Pero quiero usar algo con un poco más de información. Así que moví la descripción del modelo para encontrar mixed7,
que es la salida de una gran cantidad de convolución que son 7 por 7. No tienes que usar esta capa y es divertido experimentar con otros. Pero con este código, voy a agarrar esa capa desde el inicio y llevarla a la salida.

![img_18.png](ims%2FW3%2Fimg_18.png)

Así que ahora vamos a definir nuestro nuevo modelo, tomando la salida de la capa mixed7 del modelo inicial, que habíamos llamado last_ouput. Esto debería parecerse exactamente a los modelos densos que creaste al comienzo de este curso. 

El código es un poco diferente, pero esta es solo una forma diferente de usar la API de capas. Comienza aplanando la entrada, que simplemente resulta ser la salida desde el inicio. Y, a continuación, agregue una capa oculta densa. Y luego su capa de salida que tiene sólo una neurona activada por un sigmoide para clasificar entre dos elementos.

A continuación, puede crear un modelo utilizando la clase abstracta Model. Y pasando en la entrada y la definición de capas que acaba de crear. Y luego lo compila como antes con un optimizador y una función de pérdida y las métricas que desea recopilar.

![img_19.png](ims%2FW3%2Fimg_19.png)

Como antes vas a aumentar las imágenes con el generador de imágenes. 

![img_20.png](ims%2FW3%2Fimg_20.png)

Entonces, como antes, podemos obtener nuestros datos de 
entrenamiento del generador fluyendo desde el directorio especificado y pasando por todas las ampliaciones

![img_21.png](ims%2FW3%2Fimg_21.png)

Y ahora podemos entrenar como antes con model.fit_generator. Voy a correr por 100 épocas.

![img_22.png](ims%2FW3%2Fimg_22.png)

Lo que es interesante si haces esto, es que terminas con otra pero una situación de sobreajuste diferente. Aquí está el gráfico de la precisión de la formación frente a la validación. Como puede ver, aunque comenzó bien, la validación está desviando de la formación de una manera muy mala. Entonces, ¿cómo arreglamos esto? Echaremos un vistazo a eso en la próxima lección.

### Using dropout
[<- Return to INDEX 3](#index-3)

Otra herramienta útil para explorar en este punto es el Dropout. 

La idea detrás de ella es eliminar un número aleatorio de neuronas en su red neuronal. Esto funciona muy bien por dos razones: La primera es que las neuronas vecinas a menudo terminan con pesos similares, lo que puede llevar a un sobreajuste, por lo que eliminar algunas al azar puede eliminar esto. La segunda es que a menudo una neurona puede sobreponderar la entrada de una neurona en la capa anterior, y puede sobreespecializarse como resultado. Por lo tanto, ¡eliminarlas puede liberar a la red neuronal de este posible mal hábito! 

Vea el magnífico vídeo de Andrew explicando los abandonos aquí: 
https://www.youtube.com/watch?v=ARq74QuavAo

### Exploring dropouts
[<- Return to INDEX 3](#index-3)

![img_23.png](ims%2FW3%2Fimg_23.png)

En la lección anterior, vimos cómo transferir el aprendizaje. Pero cuando rediseñamos las características del clasificador inicial para gatos y perros, acabamos sobreajustando de nuevo. También teníamos un aumento, pero a pesar de eso, todavía sufríamos de sobreajuste. Así que analicemos algunas formas en las que podemos evitarlo en esta lección. 

![img_24.png](ims%2FW3%2Fimg_24.png)

Esta es la precisión de nuestro conjunto de entrenamiento en comparación con nuestro conjunto de validación en 100 épocas. No es muy saludable.

![img_25.png](ims%2FW3%2Fimg_25.png)

Hay otra capa en Keras llamada dropout. Y la idea detrás de la dropout es que las capas de una red neuronal a veces pueden terminar teniendo pesos similares y, posiblemente, impactarse entre sí y provocar un sobreajuste. Con un modelo tan grande y complejo como este, eso es un riesgo. Así que, si te puedes imaginar, las capas densas pueden parecerse un poco a esto.

![img_26.png](ims%2FW3%2Fimg_26.png)

Al eliminar algunas, hacemos que se vea así. Y eso tiene el efecto de que los vecinos no se afecten demasiado entre sí y, potencialmente, eliminen el sobreacondicionamiento.

![img_27.png](ims%2FW3%2Fimg_27.png)

Entonces, ¿cómo logramos esto en código? Bueno, aquí está la definición de nuestro modelo de antes. Y aquí es donde añadimos la deserción. El parámetro está entre 0 y 1 y es la fracción de unidades a eliminar.
En este caso, estamos eliminando el 20% de nuestras neuronas.

![img_24.png](ims%2FW3%2Fimg_24.png)

A modo de comparación, este es el gráfico del entrenamiento comparado con la precisión de antes de que se añadiera el dropout. 
Si observas que la validación se aleja de la formación de esta forma con el tiempo, la verdad es que es una buena opción probar con una persona que ha abandonado los estudios.

![img_28.png](ims%2FW3%2Fimg_28.png)

Y este es el impacto de la deserción. Puede ver que es muy significativo.

### Applying Transfer Learning to Cats v Dogs (Lab 1)
[<- Return to INDEX 3](#index-3)

Ahora que hemos explorado el aprendizaje por transferencia y hemos echado un vistazo a la regularización mediante el abandono, pasemos al escenario de Gatos contra Perros en este 
cuaderno [C2_W3_Lab_1_transfer_learning.ipynb](notebooks%2FW3%2FC2_W3_Lab_1_transfer_learning.ipynb)
 a continuación.

Para más información sobre cómo congelar/bloquear capas, puede explorar la 
[documentación aquí.](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Exploring Transfer Learning with Inception
[<- Return to INDEX 3](#index-3)

![img_29.png](ims%2FW3%2Fimg_29.png)

Aquí está el libro de aprendizaje de transferencia de esta lección. Vamos a pasarlo, y cuando terminemos, puedes probarlo tú mismo. La primera celda descarga los pesos de una red de inicio preentrenada y, a continuación, crea una instancia nueva de la misma utilizando esos pesos. Extraeremos una de las capas convolucionales como nuestra capa de entrada, y luego tomaremos su salida. Llamamos a esta última salida. 

![img_30.png](ims%2FW3%2Fimg_30.png)

Ahora, configuraremos nuestro modelo tomando la última salida como entrada a él. Eso se aplanará, y luego habrá una capa densa, una deserción y una capa de salida. 

![img_31.png](ims%2FW3%2Fimg_31.png)

El entrenamiento utilizará el aumento como hemos explorado antes. Entonces podemos ver que las imágenes están siendo cargadas y segregadas en clases correctamente, 2.000 para entrenamiento, 1.000 para validación

![img_32.png](ims%2FW3%2Fimg_32.png)

Ahora empezaremos el entrenamiento. Sólo voy a hacer 20 épocas. Mantenga un ojo en las métricas de precisión de validación y precisión. Estoy acelerando el video para ahorrar un poco de tiempo. Pero como puede ver, la precisión del entrenamiento está aumentando constantemente, y la precisión de validación se está asentando aproximadamente a mediados de los 90. Para el momento en que terminamos, la precisión del entrenamiento es de alrededor del 90 por ciento, y la validación está cerca del 97 por ciento. 

![img_33.png](ims%2FW3%2Fimg_33.png)

### What have we seen so far? 3
[<- Return to INDEX 3](#index-3)

Esta semana ha aprendido un montón de grandes conceptos nuevos

Usted vio el Aprendizaje por Transferencia, y cómo se puede tomar un modelo existente, congelar muchas de sus capas para evitar que se vuelvan a entrenar, y efectivamente "recordar" las convoluciones con las que fue entrenado para adaptarse a las imágenes. 

A continuación, añadió su propia DNN debajo de esto para poder volver a entrenar en sus imágenes utilizando las convoluciones del otro modelo. 

Aprendió acerca de la regularización utilizando la deserción para hacer que su red fuera más eficiente a la hora de evitar la sobreespecialización y, por tanto, el sobreajuste.

Antes de pasar al ejercicio de esta semana, ¡hagamos una prueba rápida! 

### Week 3 Quiz
[<- Return to INDEX 3](#index-3)

1. If I put a dropout parameter of 0.2, how many nodes will I lose?

   - [X] 20% of them
   - [ ] 2% of them
   - [ ] 20% of the untrained ones
   - [ ] 2% of the untrained ones

   > Correct
   > Spot on!

2. Why is transfer learning useful?

   - [ ] Because I can use all of the data from the original training set
   - [ ] Because I can use all of the data from the original validation set
   - [X] Because I can use the features that were learned from large datasets that I may not have access to
   - [ ] Because I can use the validation metadata from large datasets that I may not have access to

   > Correct
   > Exactly!

3. How did you lock or freeze a layer from retraining?

   - [ ] tf.freeze(layer)
   - [ ] tf.layer.frozen = true
   - [ ] tf.layer.locked = true
   - [X] layer.trainable = false

   > Correct
   > Well done!

4. How do you change the number of classes the model can classify when using transfer learning? (i.e. the original model handled 1000 classes, but yours handles just 2)

   - [ ] Ignore all the classes above yours (i.e. Numbers 2 onwards if I'm just classing 2)
   - [ ] Use all classes but set their weights to 0
   - [X] When you add your DNN at the bottom of the network, you specify your output layer with the number of classes you want
   - [ ] Use dropouts to eliminate the unwanted classes

   > Correct
   > Good job!

5. Can you use Image Augmentation with Transfer Learning Models?

   - [ ] No, because you are using pre-set features
   - [X] Yes, because you are adding new layers at the bottom of the network, and you can use image augmentation when training these

   > Correct
   > Correct!

6. Why do dropouts help avoid overfitting?

   - [X] Because neighbor neurons can have similar weights, and thus can skew the final training 
   - [ ] Having less neurons speeds up training

   > Correct
   > That's right!

7. What would the symptom of a Dropout rate being set too high?

   - [X] The network would lose specialization to the effect that it would be inefficient or ineffective at learning, driving accuracy down
   - [ ] Training time would increase due to the extra calculations being required for higher dropout

   > Correct
   > Indeed!

8. Which is the correct line of code for adding Dropout of 20% of neurons using TensorFlow?

   - [ ] tf.keras.layers.Dropout(20)
   - [ ] tf.keras.layers.DropoutNeurons(20),
   - [X] tf.keras.layers.Dropout(0.2),
   - [ ] tf.keras.layers.DropoutNeurons(0.2),

   > Correct
   > You've got it!

### Week 3 Wrap up
[<- Return to INDEX 3](#index-3)

![img_34.png](ims%2FW3%2Fimg_34.png)

Enhorabuena, has llegado al final de la tercera semana. Esta semana, espero que estés tan entusiasmado como yo con algunas de las cosas que has aprendido. Cuando analizaste el aprendizaje por transferencia, te diste cuenta de que algunas de las cosas que estaban ahí, que estaban disponibles para ti, que podías tomar el modelo de otra persona, un modelo enorme que llevaba mucho tiempo y con muchos datos a partir de los cuales entrenarte, pero que pudiste aprovechar para hacer que tus modelos fueran aún mejores. Personalmente, me parece algo que es realmente inspirador y que realmente me emociona. Pero la semana que viene cambiaremos un poco de tema y estudiaremos cómo pasar de la clasificación binaria a la clasificación multiclase para completar vuestra caja de herramientas. Nos vemos allí.

### Lecture Notes Week 3
[<- Return to INDEX 3](#index-3)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C2_W3.pdf](notes%2FC2_W3.pdf)
.

## Weekly Assignment - Transfer Learning Horses vs Humans
[<- Return to INDEX 0](#index-0)

### Transfer Learning - Horses vs Humans

Esta semana su ejercicio consistirá en aplicar lo que ha aprendido sobre el Aprendizaje por Transferencia para ver si puede aumentar la precisión del entrenamiento para Caballos contra Humanos. Para evitar un sobreajuste loco, la precisión de su conjunto de validación debería rondar el 95% si lo hace bien

Su entrenamiento debería detenerse automáticamente una vez que alcance esta precisión deseada.

Utilicemos ahora el aprendizaje por transferencia para aumentar la precisión del entrenamiento de Caballos contra humanos

Complete las tareas del cuaderno de 
asignaciones [C2W3_Assignment.ipynb](notebooks%2FW3%2FC2W3_Assignment.ipynb)
 y suba aquí los archivos pertinentes para su calificación.

Este cuaderno está alojado en github por lo que para guardar cualquier cambio necesita crear una copia del mismo dentro de su Drive. Puede hacerlo haciendo clic en la pestaña `Archivo` y luego en la opción `Guardar una copia en Drive`.

Deberá cargar un archivo:

El cuaderno (un archivo con extensión .ipynb ) que se utilizará para probar las siguientes funciones:

- train_val_generators

- crear_modelo_pre_entrenado

- salida_de_la_última_capa

- create_final_model


IMPORTANTE PARA UNA CALIFICACIÓN CORRECTA:- No borre las celdas ya que incluyen metadatos importantes para la calificación- Rellene sus soluciones dentro de los espacios proporcionados. Puede añadir nuevas celdas pero éstas serán omitidas por el calificador.

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

![img.png](ims%2FW4%2Fimg.png)

En esta semana de la clase, se abordó el tema de la clasificación multiclase en el contexto del aprendizaje profundo. Se explicó que, aunque en muchas aplicaciones se puede resolver el problema de clasificación con solo dos resultados, como en la clasificación de caballos contra humanos o gatos contra perros, en ocasiones es necesario trabajar con más de dos clases, como en el juego de piedra, papel o tijera, que tiene tres clases posibles.

![img_1.png](ims%2FW4%2Fimg_1.png)

Se mencionó que los estudiantes aprenderán los detalles de cómo implementar clasificadores multiclase y cómo etiquetar automáticamente las imágenes en función de los directorios en los que se encuentran. Se destacó la importancia de comprender la diferencia de programación al trabajar con clasificadores multiclase en comparación con clasificadores binarios.

![img_2.png](ims%2FW4%2Fimg_2.png)

Además, se mencionó que los estudiantes explorarán un conjunto de datos generado por ordenador para piedra, papel o tijera, que incluye una variedad de manos de diferentes razas, géneros, tamaños y estilos de uñas. Se explicó que la generación de datos por ordenador puede ser una solución eficaz para obtener conjuntos de datos diversificados sin la necesidad de contratar modelos reales.

Se resaltó la importancia de utilizar gráficos por ordenador para generar datos, ya que la tecnología ha avanzado lo suficiente como para que sea una opción viable. Se mencionó que las empresas también están explorando estas ideas para crear conjuntos de datos en diversos entornos.

![img_3.png](ims%2FW4%2Fimg_3.png)

En resumen, esta semana los estudiantes tendrán la oportunidad de trabajar con un conjunto de datos generado por ordenador y aprenderán a implementar clasificadores multiclase en TensorFlow. El objetivo es proporcionar más datos para alimentar algoritmos de aprendizaje profundo y mejorar su capacidad de clasificación en múltiples categorías.

### Moving from binary to multi-class classification
[<- Return to INDEX 4](#index-4)

![img_4.png](ims%2FW4%2Fimg_4.png)

En las últimas lecciones, has estado creando un clasificador binario. Uno que detecta dos tipos diferentes de objetos, caballo o humano, gato o perro, ese tipo de cosas. 

![img_5.png](ims%2FW4%2Fimg_5.png)

En esta lección, veremos cómo podemos extender eso a varias clases. Recuerda que cuando clasificábamos caballos o humanos, teníamos una estructura de archivos como esta. Había subdirectorios para cada clase, donde en este caso solo teníamos dos. Lo primero que tendrás que hacer es replicar esto para varias clases como esta. 

![img_6.png](ims%2FW4%2Fimg_6.png)

Es muy similar y aquí puedes ver que tanto el entrenamiento como la validación tienen tres subdirectorios. Uno para piedra, otro para papel y otro para tijeras. En estos, podemos poner imágenes de entrenamiento y validación para piedra, papel y tijera.

### Introducing the Rock-Paper-Scissors dataset
[<- Return to INDEX 4](#index-4)

Piedra-Papel-Tijeras es un conjunto de datos que contiene 2.892 imágenes de diversas manos en poses de Piedra/Papel/Tijeras. Tiene licencia
[CC By 2.](https://creativecommons.org/licenses/by/2.0/)
0 y está disponible para todos los fines, pero su intención es principalmente el aprendizaje y la investigación.

Piedra, Papel o Tijeras contiene imágenes de diversas manos, de diferentes razas, edades y géneros, en poses de Piedra, Papel o Tijeras y etiquetadas como tales. Puede descargar el
[conjunto de entrenamiento aquí](https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip)
, y el
[conjunto de prueba aquí](https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip)
. Todas estas imágenes se han generado utilizando técnicas CGI como experimento para determinar si un conjunto de datos basado en CGI puede utilizarse para la clasificación frente a imágenes reales. También he generado algunas imágenes que puede utilizar para realizar predicciones. Puede encontrarlas
[aquí](https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-validation.zip)
.

Tenga en cuenta que todos estos datos están posados sobre un fondo blanco.

Cada imagen es de 300×300 píxeles en color de 24 bits


Verá cómo se puede utilizar este conjunto de datos para construir un clasificador multiclase en el siguiente vídeo.

### Explore multi-class with Rock Paper Scissors dataset
[<- Return to INDEX 4](#index-4)

![img_7.png](ims%2FW4%2Fimg_7.png)

Ahora, este es un nuevo conjunto de datos que creé para oportunidades de aprendizaje. Está disponible gratuitamente y consta de unas 3.000 imágenes. Todas han sido generadas mediante CGI con una amplia gama de modelos, hombres y mujeres, y muchos tonos de piel diferentes. He aquí algunos ejemplos. 

![img_8.png](ims%2FW4%2Fimg_8.png)

Si quieres descargar los conjuntos de datos, puedes encontrarlos en esta URL. Contendrá un conjunto de entrenamiento, un conjunto de validación y algunas imágenes adicionales que podrás descargar para probar la red por ti mismo.

![img_9.png](ims%2FW4%2Fimg_9.png)

Una vez que tu directorio esté configurado, tendrás que configurar tu generador de imágenes. Este es el código que usaste anteriormente, pero ten en cuenta que el modo de clase estaba configurado en binario. 

![img_10.png](ims%2FW4%2Fimg_10.png)

Para varias clases, tendrás que cambiarlo a un formato categórico como este. 

> Nota importante, también se podría usar "sparse" en lugar de "categorical"

1. **Categorical:**
   
   - En este caso, `class_mode='categorical'` significa que las etiquetas serán convertidas a forma categórica, es decir, one-hot encoded. Si tienes `n` clases, cada etiqueta será un vector de `n` elementos en donde todos los elementos son `0` excepto en la posición que representa la clase a la que pertenece la imagen, que será `1`.
   - Por ejemplo, si tienes tres clases y una imagen pertenece a la clase 2, su etiqueta sería `[0, 1, 0]`.
   - Este modo es el que debes usar si tu capa de salida en el modelo de red neuronal es una capa `softmax` y estás utilizando una función de pérdida como `categorical_crossentropy`.

2. **Sparse:**
   
   - Usando `class_mode='sparse'`, las etiquetas se proporcionan como enteros. En lugar de convertir la etiqueta de la clase a un vector one-hot, la etiqueta es simplemente el índice entero de la clase.
   - Por ejemplo, si la imagen pertenece a la clase 2, la etiqueta sería simplemente `2`.
   - Este formato es útil cuando tienes muchas clases, lo cual podría hacer que la representación one-hot sea muy grande y poco eficiente. También es la representación que debes elegir si la capa de salida de tu red es un nodo singular con activación `softmax` y utilizas la función de pérdida `sparse_categorical_crossentropy`.
   
Ambos modos `categorical` y `sparse` son usados en problemas de clasificación multiclase, y la elección entre uno y otro a menudo depende del tamaño del conjunto de datos, el número de clases y también de la estructura de tu red neuronal (en particular la capa de salida y la función de pérdida).

![img_11.png](ims%2FW4%2Fimg_11.png)

El siguiente cambio se produce en la definición de tu modelo , donde tendrás que cambiar la capa de salida. Para un clasificador binario, era más eficiente tener solo una neurona y usar una función sigmoidea para activarla. Esto significaba que arrojaría un resultado cercano a cero para una clase y cercano a uno para la otra.

![img_12.png](ims%2FW4%2Fimg_12.png)

Ahora bien, eso no es adecuado para varias clases, así que tenemos que cambiarlo, pero es bastante simple. Ahora tenemos una capa de salida que tiene tres neuronas, una para cada una de las clases piedra, papel y tijera, y la activa softmax, que convierte todos los valores en probabilidades que sumarán una. 

![img_13.png](ims%2FW4%2Fimg_13.png)

Entonces, ¿qué significa eso realmente? Considera una mano como esta. Lo más probable es que sea un papel , pero
como tiene los dos primeros dedos abiertos y los demás unidos, también podría confundirse con unas tijeras. El resultado de una red neuronal con tres neuronas y un softmax reflejaría eso, y tal vez se vería así con una probabilidad muy baja de piedra, una muy alta para el papel y otra decente para las tijeras. Las tres probabilidades seguirían sumando una. 

![img_14.png](ims%2FW4%2Fimg_14.png)

El cambio final se produce entonces cuando se compila la red. Si recuerda los ejemplos anteriores, su función de pérdida era la entropía cruzada binaria. 

![img_15.png](ims%2FW4%2Fimg_15.png)

hora cambiarás si es una entropía cruzada categórica como esta. Hay otras funciones de pérdida categórica, incluida la entropía cruzada, categórica dispersa, que utilizaste en el ejemplo de moda y, por supuesto, también puedes usarlas.

![img_16.png](ims%2FW4%2Fimg_16.png)

En torno a esto durante 100 épocas, y tengo este gráfico, muestra que el entrenamiento alcanza un máximo en unas 25 épocas. Así que te recomendaría no usar muchos, y eso es todo lo que tienes que hacer. Así que echémosle un vistazo en el cuaderno de ejercicios.

### Check out the code! (Lab 1)
[<- Return to INDEX 4](#index-4)

Ahora que ya ha explorado los datos y ha visto cómo se puede modificar el código para pasar de un clasificador de clase binario a uno multiclase, ¡echemos un vistazo al 
cuaderno [C2_W4_Lab_1_multi_class_classifier.ipynb](notebooks%2FW4%2FC2_W4_Lab_1_multi_class_classifier.ipynb)
 que lo implementa por nosotros!

### Train a classifier with Rock Paper Scissors
[<- Return to INDEX 4](#index-4)

![img_17.png](ims%2FW4%2Fimg_17.png)

En la lección anterior, analizamos cómo pasaría de los clasificadores binarios que habíamos estado viendo a lo largo del curso, para actualizarlo desde el clasificador multiclase. En este vídeo, veremos el libro de trabajo de las tijeras de papel de roca y exploraremos cómo se realiza la clasificación multiclase. 


El primer paso es obtener los datos. Hay dos archivos zip, uno para los datos de entrenamiento y otro para el conjunto de pruebas. Una vez que tengamos los datos, los descomprimiremos en el directorio temporal, y los datos de entrenamiento se encuentran en barra inclinada temp rps. 

```python
# Download the train set
!wget https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip

# Download the test set
!wget https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip

import zipfile

# Extract the archive
local_zip = './rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/rps-train')
zip_ref.close()

local_zip = './rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/rps-test')
zip_ref.close()

```
Puede ver los subdirectorios aquí. Ahora echemos un vistazo a algunos de los archivos dentro de eso. 
```python
import os

base_dir = 'tmp/rps-train/rps'

rock_dir = os.path.join(base_dir, 'rock')
paper_dir = os.path.join(base_dir, 'paper')
scissors_dir = os.path.join(base_dir, 'scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])
```
Puede ver que hay 840 de cada clase y algunos nombres de archivo. Vamos a trazar algunos de los archivos para que podamos ver cómo se ven los datos.
```commandline
total training rock images: 840
total training paper images: 840
total training scissors images: 840
['rock06ck02-074.png', 'rock02-012.png', 'rock01-097.png', 'rock03-105.png', 'rock05ck01-058.png', 'rock02-038.png', 'rock05ck01-065.png', 'rock05ck01-047.png', 'rock01-026.png', 'rock02-044.png']
['paper07-001.png', 'paper03-046.png', 'paper05-110.png', 'paper07-056.png', 'paper03-087.png', 'paper03-015.png', 'paper03-019.png', 'paper07-115.png', 'paper01-000.png', 'paper01-058.png']
['scissors01-052.png', 'testscissors02-089.png', 'testscissors01-069.png', 'testscissors03-011.png', 'scissors01-023.png', 'scissors04-070.png', 'testscissors02-104.png', 'scissors02-019.png', 'testscissors03-103.png', 'testscissors02-034.png']
```

```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname)
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname)
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname)
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()
```


![img_18.png](ims%2FW4%2Fimg_18.png)

Como podemos ver, tenemos algunas manos diferentes con diferentes colores de piel, y manos masculinas y femeninas. 


Ahora vamos a construir el modelo. Tenga en cuenta que aunque las imágenes son de 300 por 300, estamos configurando los generadores de imágenes para darnos una varianza de 150 por 150. Los redimensionará sobre la marcha y aumentará los que están en el directorio de entrenamiento.

```python
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
```

Ahora, comienza el entrenamiento. Voy a acelerar el video para que puedas ver el progreso. Mantenga un ojo en la precisión y la precisión de validación

```python
# Set the training parameters
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAINING_DIR = "tmp/rps-train/rps"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
        rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "tmp/rps-test/rps-test-set"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150,150),
    class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150,150),
    class_mode='categorical',
  batch_size=126
)

# Train the model
history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)
```

ólo estoy entrenando durante 25 épocas, basado en el gráfico que viste en la última lección. Pero para cuando lleguemos a la décima época, ya lo estamos haciendo bastante bien. Para cuando terminemos, los datos de entrenamiento están por encima del 98% y los datos de validación tienen una precisión del 95%.

```commandline
Epoch 1/25
20/20 [==============================] - 31s 1s/step - loss: 1.1151 - accuracy: 0.3337 - val_loss: 1.0972 - val_accuracy: 0.3333
Epoch 2/25
20/20 [==============================] - 22s 1s/step - loss: 1.1024 - accuracy: 0.3738 - val_loss: 1.1202 - val_accuracy: 0.3333
Epoch 3/25
20/20 [==============================] - 23s 1s/step - loss: 1.1008 - accuracy: 0.3329 - val_loss: 1.0950 - val_accuracy: 0.3656
...
Epoch 23/25
20/20 [==============================] - 22s 1s/step - loss: 0.1311 - accuracy: 0.9548 - val_loss: 0.1216 - val_accuracy: 0.9570
Epoch 24/25
20/20 [==============================] - 21s 1s/step - loss: 0.2298 - accuracy: 0.9254 - val_loss: 0.0379 - val_accuracy: 0.9946
Epoch 25/25
20/20 [==============================] - 22s 1s/step - loss: 0.0767 - accuracy: 0.9766 - val_loss: 0.0408 - val_accuracy: 0.9785
```
Se trata de datos altamente especializados que están optimizados para esta lección y no un gran escenario real para Rock, Paper y Tijeras. Discutiré por qué en un momento , pero primero vamos a trazar la precisión. 

```python
import matplotlib.pyplot as plt

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
```

![img_19.png](ims%2FW4%2Fimg_19.png)

Podemos ver el entrenamiento mejorando y la tendencia hacia uno. La validación en zig-zags un poco, pero siempre está entre 0.9 y una después de las primeras épocas. Ahora, vamos a explorar probándolo con algunas imágenes que no ha visto anteriormente.


### Try testing the classifier
[<- Return to INDEX 4](#index-4)

En el siguiente vídeo, verá a Laurence probando el clasificador. También debería probarlo utilizando los datos que puede encontrar
[aquí](https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps-test-set.zip)
.

### Test the Rock Paper Scissors classifier
[<- Return to INDEX 4](#index-4)

![img_20.png](ims%2FW4%2Fimg_20.png)

He generado algunos que puedes usar para probar, están en la URL en la lección anterior con el conjunto de entrenamiento, el conjunto de pruebas y el conjunto de validación están todos vinculados a. Descarga el zip de validación de RPS desde el enlace que se muestra y en él encontrarás 33 imágenes de diferentes modelos de mano que no utilizaban los conjuntos de entrenamiento y validación y que están en diferentes poses.

![img_21.png](ims%2FW4%2Fimg_21.png)

Ahora vamos a subir uno de estos al libro de trabajo y vamos a tratar de obtener lo que sucede. Voy a elegir una pose de papel,
y la clasificación vuelve como 1-0-0. Cuando se utiliza el generador de imágenes, las clases provienen de directorios y por lo tanto se ordenan en orden alfabético. Así que el primer valor es para el papel y luego la roca y luego las tijeras.

![img_22.png](ims%2FW4%2Fimg_22.png)

También puedo cargar todos los archivos a la vez y ver clasificaciones para todos ellos. Así que intentemos eso. Estoy acelerando el video para superar todas las subidas. Así que echemos un vistazo a las clasificaciones que recibo. 

![img_23.png](ims%2FW4%2Fimg_23.png)

Recuerde que las neuronas están en el papel de orden y luego roca luego tijeras porque es alfabético. Así que podemos ver que el primero tiene razón. Es tijeras. Pero la segunda está equivocada. Es papel. Pero la tercera neurona tiene el valor más grande que está cerca de uno. Hay una pequeña señal de papel, pero no lo suficiente. El tercero es tijeras y eso es correcto y lo mismo con el cuarto y el quinto. Así que sigamos revisando la lista.

### What have we seen so far? 4
[<- Return to INDEX 4](#index-4)

Está llegando al final del Curso 2, ¡y ha recorrido un largo camino! Desde los primeros principios en la comprensión de cómo funciona el ML, hasta el uso de una DNN para hacer visión por ordenador básica, y luego más allá en convoluciones.

Con las convoluciones, usted vio cómo extraer características de una imagen, y vio las herramientas en TensorFlow y Keras para construir con convoluciones y pooling, así como el manejo de imágenes complejas y de múltiples tamaños.

A través de esto, usted vio cómo el sobreajuste puede tener un impacto en sus clasificadores, y exploró algunas estrategias para evitarlo, incluyendo el aumento de la imagen, el abandono, el aprendizaje por transferencia, y más. Para terminar, ¡examinó las consideraciones de su código para construir un modelo de clasificación multiclase! 

### Week 4 Quiz
[<- Return to INDEX 4](#index-4)

1. The diagram for traditional programming had Rules and Data In, but what came out?

   - [X] Answers
   - [ ] Binary
   - [ ] Machine Learning
   - [ ] Bugs

   > Correct
   > Exactly! Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.

2. Why does the DNN for Fashion MNIST have 10 output neurons?

   - [ ] To make it train 10x faster
   - [ ] To make it classify 10x faster
   - [ ] Purely Arbitrary
   - [X] The dataset has 10 classes

   > Correct
   > Exactly! There are 10 output neurons because we have 10 classes of clothing in the dataset. These should always match.

3. What is a Convolution?

   - [ ] A technique to make images smaller
   - [ ] A technique to make images larger
   - [X] A technique to extract features from an image
   - [ ] A technique to remove unwanted images

   > Correct

4. Applying Convolutions on top of a DNN will have what impact on training?

   - [ ] It will be slower
   - [ ] It will be faster
   - [ ] There will be no impact
   - [X] It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!

   > Correct
   > Exactly!

5. What method on an ImageGenerator is used to normalize the image?

   - [ ] normalize
   - [ ] flatten
   - [ ] rezize()
   - [X] rescale

   > Correct
   > Correct!

6. When using Image Augmentation with the ImageDataGenerator, what happens to your raw image data on-disk.

   - [ ] A copy will be made, and the copies are augmented
   - [ ] A copy will be made, and the originals will be augmented
   - [X] Nothing
   - [ ] The images will be edited on disk, so be sure to have a backup

   > Correct
   > That is, in fact, true. Nothing happens.

7. Can you use Image augmentation with Transfer Learning?

   - [ ] No - because the layers are frozen so they can't be augmented
   - [X] Yes. It's pre-trained layers that are frozen. So you can augment your images as you train the bottom layers of the DNN with them

   > Correct
   > You've got it!

8. When training for multiple classes what is the Class Mode for Image Augmentation?

   - [ ] class_mode='multiple'
   - [ ] class_mode='non_binary'
   - [X] class_mode='categorical'
   - [ ] class_mode='all'

   > Correct
   > Nicely done!


### Lecture Notes Week 4
[<- Return to INDEX 4](#index-4)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C2_W4.pdf](notes%2FC2_W4.pdf)
.

### IMPORTANT Reminder about end of access to Lab Notebooks
[<- Return to INDEX 4](#index-4)

¡Hola alumno! 

El(los) siguiente(s) tema(s) es(son) el(los) último(s) tema(s) calificado(s) de este curso. Si lo/los aprueba y también ha aprobado todos los ítems calificados anteriores, podrá obtener el certificado del curso. ¡Enhorabuena!

Tenga en cuenta que, de acuerdo con 
[la Política de Pagos y Reembolsos de Coursera](https://www.coursera.org/about/terms#payments-and-refund-policy)
, su acceso a los ítems calificados (así como a los Laboratorios No Calificados alojados en Coursera) finalizará una vez que expire su suscripción. Si desea conservar sus cuadernos de laboratorio como referencia, le recomendamos encarecidamente que 
[los descargue](https://community.deeplearning.ai/t/downloading-your-notebook-downloading-your-workspace-and-refreshing-your-workspace/475495)
 antes de finalizar el curso o la Specializations.

Puede dirigirse al 
[Centro de Ayuda al Aprendiz de Coursera](https://www.coursera.support/s/article/360036160591-How-to-contact-Coursera?language=en_US)
 si tiene preguntas sobre esta política, o si se queda bloqueado mientras tiene una suscripción activa. 

Gracias y ¡siga aprendiendo!
Equipo de control de calidad de DeepLearning.AI

### Wrap up 4
[<- Return to INDEX 4](#index-4)

¡Enhorabuena por haber terminado el Curso 2 de esta Specializations! Aquí ha pasado mucho tiempo repasando las Redes Neuronales Convolucionales: 

- explorando cómo utilizarlas con grandes conjuntos de datos

- aprovechando el aumento, el dropout, la regularización y el aprendizaje por transferencia

- y, por supuesto, examinando las consideraciones de codificación entre la clasificación binaria o multiclase

En el Curso 3, cambiará de marcha y continuará con el Procesamiento del Lenguaje Natural y, en particular, con el trabajo con textos. 

### A conversation with Andrew Ng 5
[<- Return to INDEX 4](#index-4)

![img_24.png](ims%2FW4%2Fimg_24.png)

Enhorabuena por haber llegado al final de este segundo curso. Solo porque has aprendido cómo aumentar tu confianza, cómo implementar el aumento 
de datos, cómo implementar el aprendizaje por transferencia y cómo usar el tamaño cruzado de varias clases, pero aún te queda mucho 
por aprender. Muchos de los ejemplos que has visto hasta ahora han utilizado mucha visión artificial. 

En el próximo curso, aprenderás cómo manejar el procesamiento del lenguaje natural y cómo trabajar con textos. También va a ser muy divertido cambiar de tema, de trabajar con píxeles a trabajar con personajes y palabras. Vamos a ver cómo convertir palabras en símbolos, cómo generar incrustaciones, para que podamos aprender de ellas.

![img_25.png](ims%2FW4%2Fimg_25.png)

Una incrustación es cuando podemos convertir una palabra básicamente en un vector en un espacio multidimensional y, desde la dirección en la que apunta ese vector, podemos empezar a determinar la semántica de esa palabra. Vamos a analizar todo eso y cómo las palabras también funcionan en secuencia y en diferentes modelos de secuencia para aprender cuál es el contexto de una oración y cuál es la semántica de la oración, sí.

Eso suena emocionante. Así que creo que el procesamiento del lenguaje natural realmente está despegando, en parte, debido al aprendizaje profundo. Así que en el próximo curso aprenderás mucho sobre eso y tú mismo crearás algunos de estos interesantes modelos. Así que pasemos al siguiente curso.

### Acknowledgments
[<- Return to INDEX 4](#index-4)

Además de los desarrolladores originales del plan de estudios, las siguientes personas hicieron contribuciones significativas a la actualización del Curso 2:

**Revisiones de los cuestionarios**

- Diego Peluffo

**Apoyo de ingeniería**

- Andrés Zarta

**Pruebas alfa y tutoría**

- Giovanni Lignarolo

- Deepthi Locanindi

- Chris Favila

**Probadores alfa**

- Nilosree Sengupta

- Paul Wilson Kolluri

- Moustafa Shomer

- Daewook Kim

**Marketing**

- Ishita Chaudary

**Producción de vídeo**

- Diego Peluffo

**Apoyo administrativo y de contenidos adicional**

- Ryan Keenan

- Lara Pheatt-Pitzer

- Inhae Koo

- Muhammad Mubashar

- Christopher Moroney

- Dani Zysman

- Emma Afflerbach


Por último, ¡nuestro más sincero agradecimiento a USTED, por inscribirse en esta Specializations y formar parte de nuestra comunidad global de estudiantes!


## Weekly Assignment - Multiclass Classification
[<- Return to INDEX 0](#index-0)

### Classification: Beyond two classes

Ahora que ha explorado los conceptos que hay detrás de pasar de la clasificación binaria a la clasificación multiclase, es el momento de realizar otro Ejercicio. En éste, utilizará el conjunto de datos de lenguaje de signos de
https://www.kaggle.com/datamunge/sign-language-mnist
, ¡e intentará construir un clasificador multiclase para reconocer el lenguaje de signos!

¡Construyamos un clasificador multiclase para reconocer el lenguaje de signos!

Complete las tareas del cuaderno de 
asignaciones
 y suba aquí los archivos pertinentes para su calificación.

> Este cuaderno está alojado en github por lo que para guardar cualquier cambio necesita crear una copia del mismo dentro de su Drive. Puede hacerlo haciendo clic en la pestaña `Archivo` y luego en la opción `Guardar una copia en Drive`.
> 

Deberá cargar un archivo:
El cuaderno (un archivo con extensión .ipynb ) que se utilizará para probar las siguientes funciones:

- parse_data_from_input

- train_val_generators

- create_model

> IMPORTANTE PARA UNA CALIFICACIÓN CORRECTA:- No borre las celdas ya que incluyen metadatos importantes para la calificación- Rellene sus soluciones dentro de los espacios proporcionados. Puede añadir nuevas celdas pero éstas serán omitidas por el calificador.

