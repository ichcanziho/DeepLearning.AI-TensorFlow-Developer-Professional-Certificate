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

Bienvenido a este curso sobre cómo pasar de los conceptos básicos a la maestría en TensorFlow. ¡Estamos emocionados de que estés aquí! En la Semana 1, recibirás una introducción suave a lo que son el Aprendizaje Automático y el Aprendizaje Profundo, y cómo ofrecen un nuevo paradigma de programación, dándote un nuevo conjunto de herramientas para abrir escenarios antes inexplorados. Todo lo que necesitas saber son algunas habilidades de programación muy básicas, y el resto lo irás aprendiendo sobre la marcha. Para comenzar, echa un vistazo al primer video, una conversación entre Andrew y Laurence que establece el tema de lo que estudiarás...

**Objetivos de aprendizaje**

- Monitorear la precisión de las predicciones de precios de viviendas.
- Analizar predicciones de precios de viviendas que provienen de una red neuronal de una sola capa.
- Usar TensorFlow para construir una red neuronal de una sola capa para ajustar modelos lineales.

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

![img.png](ims%2FW1%2Fimg.png)

En este video, se da la bienvenida a un curso de TensorFlow, desde lo básico hasta lo avanzado. El presentador, Laurence Moroney, 
un desarrollador en Google, comparte su experiencia en inteligencia artificial (AI) y TensorFlow. 

Destacan las herramientas como TensorFlow, PyTorch y Caffe para implementar algoritmos de aprendizaje profundo y aprendizaje automático. 
Laurence Moroney, agradecido por aprender de cursos anteriores, discute la importancia del aprendizaje automático en diversas 
industrias y la escasez de desarrolladores AI. 

El objetivo es capacitar a más personas en AI para aprovechar nuevas oportunidades. El video promete explorar diferencias 
entre paradigmas de programación y adaptar datos en el contexto de aprendizaje automático.

### Where to find the notebooks for this course
[<- Return to INDEX 1](#index-1)

Todos los cuadernos de este curso pueden ejecutarse en Google Colab o en Coursera Labs. **No necesita tener configurado un entorno local para seguir los ejercicios de codificación**. 
Puede simplemente hacer clic en la insignia `Open in Colab` en la parte superior de los laboratorios no calificados, mientras 
que para las tareas, se le llevará automáticamente a Coursera Labs. 

Sin embargo, si desea ejecutarlos en su máquina local, los laboratorios no calificados y las asignaciones para cada semana se pueden encontrar en este 
[repositorio de Github](https://github.com/https-deeplearning-ai/tensorflow-1-public)
 bajo la carpeta `C1`. Si ya tiene git instalado en su ordenador, puede clonarlo con este comando:

git clone https://github.com/https-deeplearning-ai/tensorflow-1-public

Si no, por favor siga las guías 
[aquí](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
 para instalar git en su sistema operativo. Una vez que haya clonado el repositorio, puede hacer un `git pull` de vez en 
cuando para asegurarse de que recibe las últimas actualizaciones de los cuadernos.

Necesitará estos paquetes si va a ejecutar los cuadernos localmente:

```commandline
tensorflow==2.7.0
scikit-learn==1.0.1
pandas==1.1.5
matplotlib==3.2.2
seaborn==0.11.2
```

### A primer in machine learning
[<- Return to INDEX 1](#index-1)

![img_1.png](ims%2FW1%2Fimg_1.png)

La programación ha sido el pan de cada día para los desarrolladores desde los albores de la informática. Estamos acostumbrados a crear aplicaciones descomponiendo los requisitos en problemas parciales que entonces pueden ser programados. 

Por ejemplo, si tenemos que escribir una aplicación que resuelva un analítico de valores, tal vez el precio dividido por la relación, generalmente podemos escribir un programa para obtener los valores a partir de una fuente de datos, hacer el cálculo y devolver el resultado.

![img_2.png](ims%2FW1%2Fimg_2.png)

O si estamos escribiendo un juego generalmente podemos seguir las reglas. Por ejemplo, si la bola golpea el ladrillo entonces el ladrillo debe desaparecer y la bola debe rebotar. Pero si la pelota cae fuera de la parte inferior de la pantalla entonces tal vez el jugador pierde su vida. 

![img_3.png](ims%2FW1%2Fimg_3.png)

Podemos representarlo con este diagrama. Las reglas y los datos entran y las respuestas salen. Las reglas se expresan en un lenguaje de programación, los datos pueden provenir de una variedad de fuentes de variables locales hasta de bases de datos. 

![img_4.png](ims%2FW1%2Fimg_4.png)

El aprendizaje automático reorganiza este diagrama aquí ponemos respuestas y datos entrando y salen las reglas. En lugar de ser nosotros, como desarrolladores, los que averiguamos las reglas de cuándo debe desaparecer el ladrillo, cuándo debe terminar la vida del jugador, o cuál es el analítico deseado para cualquier otro concepto, lo que haremos es conseguir un montón de ejemplos de lo que queremos ver y entonces el computador averigua las reglas. 

![img_5.png](ims%2FW1%2Fimg_5.png)

Si estoy construyendo un dispositivo que detecta si alguien, digamos camina y tengo datos de su velocidad, Podría escribir un código como este y si está corriendo, a una velocidad más rápida podría adaptar mi código a esto y si está montando en bicicleta, eso tampoco es tan malo. Puedo adaptar mi código así. Pero luego tengo que hacer reconocimiento de golf también y mi concepto se viene abajo.

Pero no sólo eso, hacerlo solamente por velocidad por supuesto es bastante ingenuo. Caminamos y corremos a diferentes velocidades cuesta arriba y cuesta abajo y otras personas caminan y corren a velocidades diferentes a nosotros. 

![img_4.png](ims%2FW1%2Fimg_4.png)

Volvamos a este diagrama. En definitiva el aprendizaje automático es muy similar, pero sólo estamos volteando los ejes. En vez de tratar de expresar el problema como reglas cuando a menudo eso ni siquiera es posible, Tengo que llegar a un compromiso. El nuevo paradigma es que tengo montones y montones de ejemplos y tengo etiquetas de esos ejemplos y uso los datos para decir esto es caminar, esto es correr, esto es andar en bicicleta y sí, incluso esto es jugar al golf. Entonces se convierte en respuestas y datos entrando con reglas de siendo inferidas por la máquina. 

![img_6.png](ims%2FW1%2Fimg_6.png)

 Un algoritmo de aprendizaje automático averigua los patrones específicos de cada conjunto de datos que determina el carácter distintivo de cada uno. Eso es lo que es tan poderoso y emocionante sobre este paradigma de programación. Es más que solo una nueva forma de hacer lo mismo de siempre. Abre nuevas posibilidades que no eran viables de hacer antes. 


### The Hello World of neural networks
[<- Return to INDEX 1](#index-1)

![img_7.png](ims%2FW1%2Fimg_7.png)

Anteriormente mencionamos que el aprendizaje automático trata de un computador aprendiendo patrones que distinguen cosas. Como para el reconocimiento de actividades era el patrón caminar, correr y andar en bicicleta que pueden ser aprendidos a partir de varios sensores de un equipo.

![img_8.png](ims%2FW1%2Fimg_8.png)

Para demostrar cómo funciona veamos un conjunto de números y veamos si puedes determinar un patrón entre ellos. Estos son los números. Existe una fórmula que mapea X en Y. ¿La puedes ver? Tómate un momento.

La respuesta es Y es igual a 2X menos 1. Siempre que veas un Y, es el doble del X correspondiente menos 1. Si lo averiguaste por ti mismo, bien hecho. ¿Pero cómo lo hiciste? ¿Cómo piensas que podrías averiguarlo? Tal vez puedes ver que Y incrementa por 2 cada vez que X incrementa por uno. Probablemente parece que Y es igual a 2X más o menos algo. Entonces cuando vistes X igual a 0 e Y igual a -1, así que pensaste que ese algo es menos 1, la respuesta puede ser 2X menos 1

![img_9.png](ims%2FW1%2Fimg_9.png)

Veámoslo ahora en código. Esta es nuestra primera línea de código. Esto está escrito usando Python y TensorFlow y un API en TensorFlow llamado Keras. Keras hace muy fácil definir redes neuronales. Una red neuronal es básicamente un conjunto de funciones que pueden aprender patrones. No te preocupes si hay muchos conceptos nuevos. Se aclararán bastante rápido a medida que trabajes con ellos.

La red neuronal más simple posible es aquella que tiene una solo neurona en ella, y eso es lo que esta línea de código hace. En Keras usas la palabra dense para definir una capa de neuronas conectadas. Aquí solo hay un dense. Así que solo hay una capa y solo hay una unidad en ella, es una sola neurona. Las sucesivas capas están definidas en secuencia, de aquí la palabra sequential. Pero como dije, solo hay una. Solo tienes una neurona

Defines la forma de la entrada a la red neuronal en la primera y en este caso única capa, puedes ver que nuestra entrada es super simple. Es solamente un valor. Probablemente has visto que, para el aprendizaje automático, necesitas saber y usar un montón de matemáticas, cálculo, probabilidades y cosas similares. Es realmente muy bueno entenderlo al querer optimizar los modelos, pero lo bueno por ahora sobre TensorFlow y Keras es que muchas matemáticas están implementadas para ti en funciones.

![img_10.png](ims%2FW1%2Fimg_10.png)

Existen dos roles de funciones de los que debes ser consciente estas son las funciones de pérdida y los optimizadores. Este código las define. Me gusta pensar en ello de esta manera. La red neuronal no tiene idea de la relación entre X e Y, así que hace una conjetura. Digamos que conjetura que Y es igual a 10X menos 10. Usará los datos que conoce que son los conjuntos de X's e Y's que ya hemos visto para medir lo buena o lo mala que fue la conjetura. 

La función de pérdida mide esto y luego pasa los datos al optimizador que averigua la siguiente conjetura. El optimizador piensa sobre lo bien o mal que se hizo la conjetura usando los datos con la función de pérdida. La lógica es que cada conjetura debe ser mejor que la anterior. A medida que las conjeturas se mejoran la precisión se acerca al 100 por ciento, se usa el término convergencia. 

En este caso, la pérdida es el error cuadrático medio y el optimizador es SGD que significa descenso de gradiente estocástico. Si deseas aprender más sobre estas funciones particulares así como las otras opciones que pueden ser mejores en otros escenarios, revisa la documentación de TensorFlow. 

![img_11.png](ims%2FW1%2Fimg_11.png)

Por ahora solo vamos a usar esto. Nuestro siguiente paso es representar los datos conocidos. Estos son los X's e Y's que viste anteriormente. El np.array usa una librería Python llamada numpy que hace la representación de datos particularmente las listas mucho más fácil. Aquí puedes ver que tenemos una lista para las X y otra para las Ys.

![img_12.png](ims%2FW1%2Fimg_12.png)

El entrenamiento se da con el comando fit. Aquí pedimos al modelo que averigüe cómo ajustar los valores de X a los valores de Y. Los epochs son igual a 500 y significa que vamos a hacer un bucle de entrenamiento 500 veces. Este bucle de entrenamiento es lo que describimos anteriormente. 

Haz una conjetura, mide lo buena o mala que es la conjetura con la función de pérdida, luego usas el optimizador y los datos para hacer otra conjetura y lo repites. Cuando el modelo ha finalizado el entrenamiento te dará valores usando el método de predicción. 

![img_13.png](ims%2FW1%2Fimg_13.png)

No has visto previamente 10, ¿y qué piensas que dará cuando le pases un 10? Puedes pensar que dará 19 porque después de todo Y es igual a 2X menos 1, y piensas que debe ser 19. Pero cuando pruebes esto en el cuaderno tú mismo, verás que da un valor muy cercano a 19 pero no exactamente 19.

¿Por qué crees que eso es así? Fundamentalmente hay dos razones principales. La primera es que lo entrenaste usando muy pocos datos. Solamente hay seis puntos. Esos seis puntos son lineales pero no hay garantía de que por cada X, la relación será Y igual 2X menos 1. Hay una gran probabilidad de que Y sea igual a 19 para X igual a 10, pero la red neuronal no es positiva. Averiguará un valor realista para Y.

Esa es la segunda razón principal. Cuando se usan redes neuronales, como ellas averiguan las respuestas para todo, ellas tratan con probabilidades. Verás eso mucho y tendrás que ver cómo manejar las respuestas para el ajuste. Tenlo en cuenta mientras trabajas con el código. 

### Intake Survey
[<- Return to INDEX 1](#index-1)

Nos encantaría saber más sobre su experiencia con DeepLearning.AI y con la IA en general. ¡Por favor, considere completar la breve encuesta de 3 preguntas para que podamos servirle mejor!

Este course utiliza una aplicación de terceros, Encuesta de admisión, para mejorar tu experiencia de aprendizaje. La aplicación hará referencia a información básica, como tu nombre, correo electrónico e ID de Coursera.

### Join our community
[<- Return to INDEX 1](#index-1)

¡Hola!

Hemos creado una comunidad para que usted pueda:

- Pedir ayuda sobre las tareas y otros contenidos del curso.

- Discutir temas del curso.

- Compartir sus conocimientos con otros alumnos.

- Crear su red de contactos.

- Enterarse de las novedades, eventos y concursos de DeepLearning.AI.

Para acceder a la comunidad de este curso, marque la casilla que aparece a continuación para indicar que acepta utilizar la aplicación de forma responsable y, a continuación, haga clic en elbotón"Iniciar aplicación" .

Si es nuevo en la comunidad, haga clic en el botón "Iniciar aplicación"  para crear su cuenta y acceder a nuestra comunidad.

Hemos creado esta 
[Guía](https://community.deeplearning.ai/c/faq/391)
 del usuario 
para usted
. Asegúrese de consultar las directrices comunitarias 
[del Código de Conducta ](https://community.deeplearning.ai/c/faq/code-of-conduct/392)
. ¿Tiene problemas para acceder a nuestra comunidad después de pulsar el botón "Iniciar aplicación"? Rellene este
 formulario
 para explicar su problema y nos pondremos en contacto con usted.

¡Esperamos verle pronto en nuestra comunidad!

- El equipo de DeepLearning.AI

### From rules to data
[<- Return to INDEX 1](#index-1)

En estos vídeos, se le ofreció una introducción a los conceptos y paradigmas del aprendizaje automático y el aprendizaje profundo. 
Vio que el paradigma tradicional de expresar reglas en un lenguaje de codificación puede no funcionar siempre para resolver un problema. 

Así, aplicaciones como la visión por ordenador son muy difíciles de resolver con programación basada en reglas. En cambio, si 
alimentamos un ordenador con suficientes datos que describimos (o etiquetamos) como lo que queremos que reconozca -dado que los 
ordenadores son realmente buenos procesando datos y emparejando patrones-, entonces podríamos "entrenar" potencialmente un sistema para 
resolver un problema. 

Vimos un ejemplo super sencillo de ello: ajustar números a una recta. Así que ahora, vamos a revisar un cuaderno y ejecutar el código 
que entrena una red neuronal para aprender cómo un conjunto de números forman una línea. Después, alimentaremos la red entrenada con un 
nuevo punto de datos y veremos si predice correctamente el valor esperado.

### Working through Hello World in TensorFlow and Python
[<- Return to INDEX 1](#index-1)

![img_14.png](ims%2FW1%2Fimg_14.png)

En la sección anterior, viste algunos detalles detrás del concepto y paradigmas del aprendizaje automático. Viste cómo era un cambio desde expresiones basadas en reglas usando código para obtener datos etiquetar esos datos, y luego tener una red neuronal que averigüe los patrones que generan las reglas. Viste a través de un ejemplo muy sencillo que tomaba algunos valores x e y y averiguaba la relación entre ellos.

![img_15.png](ims%2FW1%2Fimg_15.png)

Ahora vas a practicar escribiendo este código tú mismo. No necesitas un entorno de desarrollo para hacerlo, una forma en que lo puedes hacer, es usarlo correctamente en el navegador con algo llamado Google Colaboratory. Si estás familiarizado con Jupyter Notebooks en Python, estarás como en casa, y por supuesto también puedes usar Jupyter Notebooks. De otra forma, considera Colab un entorno que se ejecuta en el navegador que permite ejecutar, editar, e inspeccionar el código Python. Es realmente genial para aprender. Si desean más detalles sobre esto, revisa este vídeo en Youtube.

![img_16.png](ims%2FW1%2Fimg_16.png)

En este segmento, repaso la primera lección del curso y luego me animan a ejecutar el código por mi cuenta. Explico cómo importar y configurar TensorFlow, Keras y NumPy. Luego, defino una red neuronal con una capa, una neurona y un valor de entrada, y procedo a compilar la red utilizando una función de pérdida y un optimizador.

![img_17.png](ims%2FW1%2Fimg_17.png)

Destaco la importancia de la función de pérdida y el optimizador en ayudar a la red neuronal a aprender patrones. Luego, muestro cómo proporcionar datos conocidos (x) y desconocidos (y) usando NumPy arrays, y describo el proceso de entrenamiento de la máquina para aprender patrones a lo largo de 500 épocas.

![img_18.png](ims%2FW1%2Fimg_18.png)

Durante el entrenamiento, observo la disminución gradual de la pérdida, indicando que la red neuronal se acerca a identificar la relación entre x e y. Después de las 500 épocas, la pérdida es mínima, lo que sugiere una comprensión cercana de la relación para los datos proporcionados.

![img_19.png](ims%2FW1%2Fimg_19.png)

Finalmente, presento una nueva x (10) y muestro la predicción de la red neuronal (18.98), destacando que aunque es cercana a la respuesta correcta (19), no es exacta. Me invitan a revisar la lección anterior para comprender las razones detrás de esta discrepancia.

### Get started with Google Colaboratory (Coding TensorFlow)
[<- Return to INDEX 1](#index-1)

Puedes acceder a un breve video de [YouTube](https://www.youtube.com/watch?v=inN8seMm7UI) para conocer los principios
básicos de Cómo se utiliza Google Colab para correr código Python.

### Faqs about Google Colab
[<- Return to INDEX 1](#index-1)

Como ya se ha mencionado, puede utilizar Google Colab para seguir los ejercicios de codificación de este curso. El vídeo anterior mostraba una rápida introducción sobre cómo utilizar este entorno y aquí tiene algunas 
[preguntas frecuentes](https://research.google.com/colaboratory/faq.html)
 por si quiere saber más.

### Try it for yourself (Lab 1)
[<- Return to INDEX 1](#index-1)

Bien, ahora que ha visto la demostración, ¿por qué no lo prueba usted mismo? Puede descargar el libro de ejercicios 
aquí [C1_W1_Lab_1_hello_world_nn.ipynb](notebooks%2FC1_W1_Lab_1_hello_world_nn.ipynb)
. 

Ahora bien, aunque esto pueda parecer muy sencillo, en realidad ya ha aprendido lo básico sobre el funcionamiento de las redes neuronales. Continuará utilizando las mismas técnicas a medida que sus aplicaciones se vuelvan más complejas.

### Week 1 Quiz
[<- Return to INDEX 1](#index-1)

1. What is the difference between traditional programming and Machine Learning?

   - [X] In traditional programming, a programmer has to formulate or code rules manually, whereas, in Machine Learning, the algorithm automatically formulates the rules from the data.
   - [ ] Machine learning identifies complex activities such as golf, while traditional programming is better suited to simpler activities such as walking.

   > Correct
   > Exactly! Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.

2. What do we call the process of telling the computer what the data represents (i.e. this data is for walking, this data is for running)?

   - [ ] Learning the Data
   - [ ] Categorizing the Data
   - [X] Labelling the Data
   - [ ] Programming the Data

   > Correct
   > Yes! Labeling typically takes a set of unlabeled data and augments each piece of it with informative tags.

3. What is a Dense layer?

   - [ ] A single neuron
   - [ ] An amount of mass occupying a volume
   - [ ] A layer of disconnected neurons
   - [X] A layer of neurons fully connected to its adjacent layers

   > Correct
   > Correct! In Keras, dense is used to define this layer of connected neurons.

4. How do you measure how good the current ‘guess’ is?

   - [ ] Figuring out if you win or lose
   - [X] Using the Loss function
   - [ ] Training a neural network

   > Correct
   > Absolutely! An optimization problem seeks to minimize a loss function.

5. What does the optimizer do?

   - [ ] Measures how good the current guess is
   - [X] Generates a new and improved guess
   - [ ] Figures out how to efficiently compile your code
   - [ ] Decides to stop training a neural network

   > Correct
   > Nailed it! The optimizer figures out the next guess based on the loss function.

6. What is Convergence?

   - [ ] An analysis that corresponds too closely or exactly to a particular set of data.
   - [ ] A programming API for AI
   - [ ] A dramatic increase in loss
   - [X] The process of getting very close to the correct answer

   > Correct
   > That’s right! Convergence is when guesses get better and better closing to a 100% accuracy.

7. What does model.fit do?

   - [ ] It optimizes an existing model
   - [X] It trains the neural network to fit one set of values to another
   - [ ] It determines if your activity is good for your body
   - [ ] It makes a model fit available memory
   
   > Correct
   > Correct! The training takes place on the fit command.



### Lecture Notes 1
[<- Return to INDEX 1](#index-1)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
[C1_W1.pdf](notes%2FC1_W1.pdf)
.

NOTA: Si aún no tiene una cuenta, siga las instrucciones 
[aquí](https://www.coursera.org/learn/introduction-tensorflow/ungradedLti/3L0GK/important-have-questions-issues-or-ideas-join-our-community)
 y luego vuelva a esta página.


#### Derechos de autor
Estas diapositivas se distribuyen bajo licencia Creative Commons.

[DeepLearning.](https://www.deeplearning.ai/)
AI pone estas diapositivas a su disposición con fines educativos. No puede utilizar ni distribuir estas diapositivas con fines comerciales. Puede hacer copias de estas diapositivas y utilizarlas o distribuirlas con fines educativos siempre que cite a
[DeepLearning](https://www.deeplearning.ai/)
.AI como fuente de las mismas.

Para conocer el resto de los detalles de la licencia, consulte
https://creativecommons.org/licenses/by-sa/2.0/legalcode

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