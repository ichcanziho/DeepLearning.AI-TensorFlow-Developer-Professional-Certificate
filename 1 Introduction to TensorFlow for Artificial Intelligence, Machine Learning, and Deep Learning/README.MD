# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning

En este curso, aprenderás:

- Aprenda las mejores prácticas para utilizar TensorFlow, un popular marco de aprendizaje automático de código abierto

- Construir una red neuronal básica en TensorFlow

- Entrenar una red neuronal para una aplicación de visión por ordenador

- Entienda cómo utilizar las convoluciones para mejorar su red neuronal

Si usted es un desarrollador de software que quiere construir algoritmos escalables impulsados por IA, necesita entender cómo utilizar las herramientas para construirlos. Este curso forma parte de la próxima Especialización en Aprendizaje Automático en Tensorflow y le enseñará las mejores prácticas para utilizar TensorFlow, un popular marco de trabajo de código abierto para el aprendizaje automático. 

El curso de Aprendizaje Automático y la Especialización en Aprendizaje Profundo de Andrew Ng enseñan los principios más importantes y fundacionales del Aprendizaje Automático y el Aprendizaje Profundo. Esta nueva Especialización en TensorFlow de deeplearning.ai le enseña cómo utilizar TensorFlow para implementar esos principios, de forma que pueda empezar a construir y aplicar modelos escalables a problemas del mundo real. Para desarrollar una comprensión más profunda de cómo funcionan las redes neuronales, le recomendamos que realice la Especialización en Aprendizaje Profundo.

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
aquí [C1_W1_Lab_1_hello_world_nn.ipynb](notebooks%2FW1%2FC1_W1_Lab_1_hello_world_nn.ipynb)
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

He aquí algunas directrices generales antes de entregar sus tareas en este curso. Téngalas en cuenta no sólo para la tarea de esta semana, sino también para las siguientes:

1. Asegúrese de guardar su trabajo antes de hacer clic en el botón `Submit`. De lo contrario, es posible que aparezca un 
mensaje de error como el de la celda de código siguiente. Recuerde que todo lo que tiene que rellenar dentro de las funciones 
calificadas se inicializa en `None`.

   ```commandline
   Failed test case: x has incorrect type.
   Expected:
   some.Type,
   but got:
   <class 'NoneType'>.
   ```

2. Por favor, no cambie el nombre del cuaderno. El calificador buscará el nombre de archivo original y sus metadatos asociados, 
por lo que debe trabajar en el archivo que se abre automáticamente al hacer clic en el botón `Launch Notebook`. 
Si intenta enviar un cuaderno renombrado, es posible que también aparezca un error como el que se muestra arriba.

3. Por favor, no modifique ningún código fuera de las etiquetas `START CODE HERE` y `END CODE HERE`. Su solución sólo debe 
colocarse entre estos marcadores para garantizar una calificación correcta. Modificar los parámetros de las funciones y 
otras celdas de prueba probablemente romperá el calificador. Si desea experimentar con ellas, puede hacerlo después de haber 
superado con éxito la tarea.

4. Después de seguir los consejos anteriores y el calificador le sigue dando 0/100, es posible que los metadatos necesarios para la calificación estén dañados. Por favor, obtenga un nuevo cuaderno de laboratorio refrescando su espacio de trabajo
([instrucciones aquí](https://www.coursera.org/learn/introduction-tensorflow/supplement/M20rh/optional-downloading-your-notebook-and-refreshing-your-workspace)
). A continuación, copie sus soluciones en el nuevo cuaderno. Asegúrese de que todas las celdas siguen funcionando como se espera y, a continuación, vuelva a enviarlas.

5. Si tiene más preguntas, por favor cree un tema en la comunidad Discourse en lugar de los foros de discusión de Coursera. Puede unirse 
[siguiendo las instrucciones aquí](https://www.coursera.org/learn/introduction-tensorflow/ungradedLti/3L0GK/have-questions-join-us-on-discourse)
. Obtendrá ayuda allí más rápidamente porque varios mentores y sus compañeros de aprendizaje están supervisando los mensajes. Sólo asegúrese de crear el tema en la categoría correcta del curso.

### Downloading your Notebook and refreshing your workspace
[<- Return to INDEX 3](#index-3)

Este curso utiliza Coursera **Labs** para proporcionar un entorno de cuadernos y calificar su trabajo. Puede haber algunos casos en 
los que necesite descargar sus cuadernos o actualizar su espacio de trabajo para empezar desde cero. Este elemento de lectura 
describe los pasos para hacerlo.

#### Descarga de su cuaderno

En caso de que necesite descargar su cuaderno para solucionar problemas o para ejecutarlo en su entorno local, puede seguir estos sencillos pasos:

1. En la barra de menús del cuaderno en el que esté trabajando, haga clic en **File → Save and Checkpoint** para guardar primero su progreso.

2. Haga clic en **File → Download as → Notebook (.ipynb)**. Esto debería iniciar la descarga del archivo en su máquina local.

#### Actualizar su espacio de trabajo

Esto le resultará útil siempre que necesite empezar de cero, buscar la última versión de la tarea o encontrarse con un error 404.

1. Abra el cuaderno desde el aula.

2. Una vez abierto el cuaderno, haga clic en **File → Open**

3. Cuando se abra su espacio de trabajo, marque la casilla situada delante del archivo del cuaderno. Una vez seleccionado, pulse **Shutdown**. El icono junto al nombre del archivo debería pasar de verde a gris.

4. Marque de nuevo la casilla de verificación y esta vez elija **Rename** e introduzca cualquier nombre de archivo que no sea el original. 
Por ejemplo, **C4W1_Assignment.ipynb** (original) → **C4W1_Assignment_v2.ipynb**

5. (Opcional) Marque la casilla de cualquier otro archivo del que desee obtener una copia nueva (por ejemplo, archivos de conjuntos de datos 
que pueda haber manipulado de forma irreversible). A continuación, haga clic en **Delete**. También puede optar por **Rename** o **Download** cada archivo individualmente en caso de que desee conservarlos antes de borrarlos.

6. Haga clic en el botón **Help** situado en la parte superior derecha de la página.

7. Haga clic en el botón **Get latest version**.

8. Haga clic en el botón **Update Lab**. La página se actualizará y ahora debería ver la última versión del cuaderno.

### Housing Prices
[<- Return to INDEX 3](#index-3)

¡Genial! ¡Ya ha recorrido un largo camino! Ahora es el momento de hacer un ejercicio de programación. A principios de esta semana, 
vio un "**Hola Mundo**" en Aprendizaje Automático que predecía una relación entre los valores X e Y. Se trataba de algo 
puramente arbitrario, pero le dio la pauta de cómo puede resolver problemas más difíciles. 

Para este ejercicio, escribirá código que realice una tarea similar -- en este caso, predecir los precios de la vivienda 
basándose en una ecuación lineal simple.

> ### Nota:
> El notebook de este ejercicio es el siguiente: [C1W1_Assignment.ipynb](notebooks%2FW1%2FC1W1_Assignment.ipynb)

Para enviar su tarea para su calificación, haga clic en el botón Enviar tarea dentro del cuaderno.

> IMPORTANTE PARA UNA BUENA CALIFICACIÓN:
>
> - No olvide guardar su cuaderno antes de enviarlo
>
> - No borre las celdas, ya que incluyen metadatos importantes para la calificación.
>
> - Rellene sus soluciones dentro de los espacios proporcionados. Puede añadir nuevas celdas pero éstas serán omitidas por el calificador.

Si tiene alguna duda sobre las tareas de este curso, solicite ayuda en nuestra comunidad. Si aún no lo ha hecho, 
[haga clic aquí y siga las instrucciones para poder unirse](https://www.coursera.org/learn/introduction-tensorflow/ungradedLti/3L0GK/important-have-questions-issues-or-ideas-join-our-community-on-discourse)

### Week 1 Resources
[<- Return to INDEX 3](#index-3)

Con esto llegamos al final de lo que debe consultar en la Semana 1. Si está ansioso por aprender más, antes de pasar a la Semana 2, hay algunos recursos estupendos que puede consultar:

- AI For Everyone es un curso no técnico que le ayudará a entender muchas de las tecnologías de IA que discutiremos más adelante en este curso, y le ayudará a detectar oportunidades en la aplicación de esta tecnología para resolver sus problemas. 
https://www.deeplearning.ai/ai-for-everyone/

- TensorFlow está disponible en 
[TensorFlow.org](https://tensorflow.org/)
, y las actualizaciones en vídeo del equipo de TensorFlow están en 
[youtube.com/tensorflow](https://www.youtube.com/tensorflow)

Juegue con una red neuronal directamente en el navegador en 
http://playground.tensorflow.org.
 Vea si puede averiguar los parámetros para conseguir que la red neuronal haga coincidir los patrones con los grupos deseados. ¡La espiral es particularmente desafiante!

![img_20.png](ims%2FW1%2Fimg_20.png)

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

![img.png](ims%2FW2%2Fimg.png)

En la lección reciente, se exploraron los fundamentos del nuevo paradigma en programación que surge con el aprendizaje automático y el aprendizaje profundo. Se destacó cómo, en lugar de codificar reglas explícitas, es posible utilizar datos y conjuntos de datos etiquetados para adentrarse en campos como el reconocimiento de actividades. Añadiendo un toque de diversión, se inició la codificación de una red neuronal básica, diseñada para ajustar datos x e y en una línea. Este paso inicial, aunque sencillo, reveló una verdad asombrosa: la habilidad de ajustar relaciones entre x e y es la base para logros notables en el aprendizaje automático, como permitir que las computadoras reconozcan imágenes y realicen reconocimiento de actividades.

Se profundizó en cómo la visión por computadora representa un reto significativo. Distinguir en una imagen si se muestra un vestido o unos zapatos es una tarea compleja para una máquina. Mientras que para los humanos reconocer objetos es una tarea sencilla, para una computadora implica analizar y comprender los valores de brillo de cada píxel. Resulta fascinante que, a través del aprendizaje automático y profundo, las computadoras estén avanzando notablemente en estas capacidades.

El uso de la red neuronal básica desarrollada en la lección proporciona una plantilla para todo lo que se puede lograr con el aprendizaje profundo, diseñando redes neuronales en capas capaces de reconocer patrones complejos. Esto abre la posibilidad de aplicar estas técnicas, como se demostró, al reconocimiento de prendas de vestir. En el próximo video, se abordará cómo escribir código para aplicar este paradigma, ya introducido, en el reconocimiento de ropa a partir de datos etiquetados, invitando a continuar con el siguiente material del curso.

### An Introduction to computer vision
[<- Return to INDEX 4](#index-4)

![img_1.png](ims%2FW2%2Fimg_1.png)

En la lección anterior, vimos aprendiste qué es el paradigma de aprendizaje automático y cómo usar datos y etiquetas y hacer que una computadora infiera las reglas por ti. Viste un ejemplo sencillo donde averiguaba las relaciones entre dos conjuntos de números. Llevemos esto al siguiente nivel resolviendo un problema real, visión por computadora. 

![img_2.png](ims%2FW2%2Fimg_2.png)

Visión por computadora es el campo en el que una computadora entiende y etiqueta lo que está presente en una imagen. Considera esta diapositiva. Cuando la ves, puedes interpretar que es una camisa o que es un zapato, ¿pero cómo lo programarías? Si un extraterrestre que nunca ha visto ropa entra en el cuarto contigo, ¿Como le explicarías los zapatos? Es realmente muy difícil, sino imposible hacerlo. Y es el mismo problema con la visión por computadora. 

![img_3.png](ims%2FW2%2Fimg_3.png)

Una forma de resolverlo es utilizar un montón de imágenes de ropa decirle al computador qué es esa imagen y luego dejar que la computadora averigüe los patrones que te dan la diferencia entre un zapato una camiseta, una cartera, y un abrigo. Aprenderás cómo hacerlo en esta sección. Afortunadamente, existe un conjunto de datos llamado Fashion MNIST que da 70 mil imágenes distribuidas en 10 diferentes artículos de ropa. Estas imágenes han sido reducidas a 28 por 28 píxeles.

Usualmente, cuanto más pequeño, mejor porque la computadora debe hacer menos procesamiento. Pero por supuesto, debes retener suficiente información para estar seguro de que las características y el objeto se puedan distinguir. Si ves esta diapositiva puedes decir la diferencia entre camisetas, zapatos y carteras. Este tamaño parece ser ideal, y lo hace genial para entrenar una red neuronal. 

![img_4.png](ims%2FW2%2Fimg_4.png)

Las imágenes están también en escala gris, la cantidad de información también ha sido reducida. Cada píxel puede ser representado en valores desde cero hasta 255 y sólo es un byte por píxel. Con 28 por 28 píxeles por imagen, solo 784 bytes son necesarios para almacenar la imagen completa. A pesar de eso, podemos aún ver qué hay en la imagen y en este caso, es un botín.

### Exploring how to use data
[<- Return to INDEX 4](#index-4)

El aprendizaje automático depende de tener buenos datos con los que entrenar un sistema. En el vídeo anterior, se le propuso entrenar un sistema para reconocer imágenes de moda. Los datos proceden de un conjunto de datos llamado Fashion MNIST, y puede obtener más información sobre él en GitHub 
[aquí](https://github.com/zalandoresearch/fashion-mnist)
. En el siguiente vídeo, verá cómo cargar esos datos y prepararlos para el entrenamiento.

------------

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

![img_5.png](ims%2FW2%2Fimg_5.png)

![embedding.gif](ims%2FW2%2Fembedding.gif)

**Labels**
Each training and test example is assigned to one of the following labels:

| Label | Description    |
|-------|----------------|
| 0     | T-shirt/top    |
| 1     | Trouser        |
| 2     | Pullover       |
| 3     | Dress          |
| 4     | Coat           |
| 5     | Sandal         |
| 6     | Shirt          |
| 7     | Sneaker        |
| 8     | Bag            |
| 9     | Ankle boot     |

### Writing code to load training data
[<- Return to INDEX 4](#index-4)

![img_6.png](ims%2FW2%2Fimg_6.png)

¿Qué aspecto tendrá el manejo de esto en código? En la lección anterior, vimos aprendiste de TensorFlow y Keras, y cómo definir una red neuronal sencilla con ellos. En esta lección, las usarás para ir un poco más a fondo, pero el API en general debe parecerte familiar. La única gran diferencia, estará en los datos. La última vez tenías 6 pares de números, y podías codificarlo a mano. 

Esta vez, debes cargar 70000 imágenes del disco, habrá un poco de código para manejar eso. Afortunadamente, aún es bastante sencillo porque Fashion-MNIST está disponible como un conjunto de datos con una llamada del API de TensorFlow.

![img_7.png](ims%2FW2%2Fimg_7.png)

Simplemente declaramos un objeto de tipo MNIST cargándolo de la base de datos Keras. En este objeto, si llamamos al método de carga de datos, nos devolverá 4 listas. Los datos de entrenamiento, las etiquetas de entrenamiento, los datos de prueba y las etiquetas de prueba. 

![img_8.png](ims%2FW2%2Fimg_8.png)

¿Cuáles son estas podrías preguntar? Cuando se construye una red neuronal como esta, es una buena estrategia usar algunos de tus datos para entrenar la red neuronal y datos similares que el modelo aún no ha visto para probar lo buena que es reconociendo imágenes. En el conjunto de datos Fashion MNIST, 60.000 de las 70.000 imágenes son usadas para entrenar la red, y entonces 10.000 imágenes, unas que no ha visto previamente pueden ser usadas para probar lo bien o mal que lo está realizando.

![img_9.png](ims%2FW2%2Fimg_9.png)

Este código te dará esos conjuntos. Cada conjunto tiene datos, las imágenes mismas y las etiquetas, y eso es lo que la imagen es en realidad. Entonces, por ejemplo, los datos de entrenamiento contendrán imágenes como esta, y una etiqueta que describe la imagen así. Esta imagen es un botín, la etiqueta que la describe es el número 9. 

![img_10.png](ims%2FW2%2Fimg_10.png)

¿Por qué crees que podría ser? Existen dos razones principales. Primero, por supuesto, es que las computadoras son mejores con 
números que con textos. Segundo, muy importante, es que esto es algo que nos puede ayudar a reducir sesgos. 

Si lo etiquetáramos como un botín, por supuesto, estaríamos sesgando hacia los anglo parlantes. Pero al ser una etiqueta 
numérica, podemos referirnos a ello en nuestro propio lenguaje sea inglés, chino, japones o incluso gaélico irlandés.

### The structure of Fashion MNIST data
[<- Return to INDEX 4](#index-4)

Aquí vio cómo los datos pueden cargarse en estructuras de datos de Python que facilitan el entrenamiento de una red neuronal. Ha visto cómo la imagen se representa como una matriz de 28x28 de escalas de grises, y cómo su etiqueta es un número. Utilizar un número es un primer paso para evitar el sesgo, ¡en lugar de etiquetarla con palabras en un idioma específico y excluir a las personas que no hablan ese idioma! Puede obtener más información sobre el sesgo y las técnicas para evitarlo 
[aquí](https://developers.google.com/machine-learning/fairness-overview/)
.

1. **Prácticas Generales Recomendadas para la IA:**

   - **Diseño Centrado en el Humano:** Incluye la experiencia del usuario en la evaluación del impacto del sistema.
   - **Identificación de Múltiples Métricas:** Utilizar diversas métricas para entender los compromisos entre diferentes tipos de errores y experiencias.
   - **Examen Directo de Datos en Bruto:** Analizar cuidadosamente los datos de entrenamiento.
   - **Comprensión de las Limitaciones de Conjuntos de Datos y Modelos:** Ser consciente de que los modelos reflejan los patrones de sus datos de entrenamiento.
   - **Pruebas Rigurosas:** Asegurarse de que el sistema de IA funcione como se pretende.
   - **Monitoreo y Actualización Continua Post-Despliegue:** Tener en cuenta el rendimiento en el mundo real y el feedback de los usuarios.

2. **Ejemplos de Trabajo de Google:** Proyectos y modelos específicos que demuestran estas prácticas.

3. **Justicia en la IA:**

   - Trabajar hacia sistemas que sean justos e inclusivos para todos.
   - Reconocer los desafíos, como los sesgos preexistentes en los datos y la dificultad de construir sistemas justos en todas las situaciones o culturas.
   - No hay una definición estándar de justicia, lo que requiere un enfoque holístico y la colaboración continua en la investigación.

4. Interpretabilidad:

   - La importancia de la capacidad para cuestionar, comprender y confiar en un sistema de IA.
   - Comparación entre la toma de decisiones humana y de IA en términos de explicación y comprensión.
   - Prácticas recomendadas para mejorar la interpretabilidad, como la planificación de opciones para la interpretabilidad, el tratamiento de la interpretabilidad como parte central de la experiencia del usuario, y el diseño de modelos interpretables.

Cada sección incluye prácticas recomendadas y ejemplos específicos para ilustrar cómo Google implementa estas ideas en su trabajo con la IA. La página también enfatiza la importancia de la continua investigación y el desarrollo en estas áreas para mejorar la responsabilidad en el uso de la IA.

### Coding a Computer Vision Neural Network
[<- Return to INDEX 4](#index-4)

![img_11.png](ims%2FW2%2Fimg_11.png)

Ahora veremos el código para la definición de una red neuronal. Recuerda que la última vez teníamos sequential con una sola capa en ella.

![img_12.png](ims%2FW2%2Fimg_12.png)

Ahora tenemos 3 capas. Las cosas importantes a mirar son la primera y última capas. La última capa tiene 10 neuronas porque tenemos 10 clases de ropa en el conjunto de datos. Esto debe siempre coincidir. La primera capa es una capa flatten con la forma de entrada de 28 por 28. 

Si recuerdas, nuestras imágenes son de 28 por 28, estamos especificando que esta es la forma en que debemos esperar los datos de entrada. Flatten toma este cuadrado de 28 por 28 y lo transforma en un vector lineal simple. Lo interesante pasa en la capa intermedia, muchas veces llamada también capa oculta.

![img_13.png](ims%2FW2%2Fimg_13.png)

Esta tiene 128 neuronas me gustaría que piensen en ellas como variables en una función. Quizás x1, x2, x3, etc. Existe una regla que incorpora todas estas que convierte los 784 valores de un botín en el valor nueve, y similarmente para todos los otros 70.000. Es una función muy compleja de ver mapeando las imágenes tú mismo, pero eso es lo que hace una red neuronal. 

Por ejemplo si dices la función y es igual a w1 por x1, más w2 por x2, más w3 por x3, hasta w128 por x128. Hallando los valores de w, entonces y dará nueve, cuando tienes el valor de entrada del zapato.

![img_14.png](ims%2FW2%2Fimg_14.png)

Verás que está haciendo algo muy, muy similar a lo que hicimos anteriormente cuando hallamos y igual a 2x menos 1. En este caso el dos, era el peso de x. Estoy diciendo que y es igual a w1 por x1, etc. No te preocupes si esto no está muy claro en este momento. Con el tiempo lo dominarás, viendo que funciona, hay algunas herramientas que te permitirán mirar dentro para ver lo que está sucediendo. 

Lo importante por ahora es que el código funcione, para que veas un escenario de clasificación por ti mismo. También puedes modificar la red neuronal añadiendo, eliminando y cambiando el tamaño de la capa para ver el impacto.

![img_15.png](ims%2FW2%2Fimg_15.png)

Lo harás en el siguiente ejercicio. Asimismo, si deseas ir más allá, revisa este tutorial de Andrew en Youtube, que aclarará cómo las capas densamente conectadas funcionan desde las perspectivas teórica y matemática. Pero por ahora, volvamos al código.

https://www.youtube.com/watch?v=fXOsFF95ifk&ab_channel=DeepLearningAI

### See how it's done
[<- Return to INDEX 4](#index-4)

En el siguiente vídeo, Laurence le guiará a través de un cuaderno de trabajo en el que podrá ver cómo se entrena una red neuronal con imágenes de moda. ¡Después podrá probar el cuaderno de trabajo usted mismo!

### Walk through a Notebook for computer vision
[<- Return to INDEX 4](#index-4)

![img_16.png](ims%2FW2%2Fimg_16.png)

Acabas de ver cómo crear una red neuronal que da capacidades de visión por computadora básicas para reconocer diferentes artículos de ropa. 
Trabajemos ahora con un cuaderno que tiene todo el código para hacer eso. Luego verás el cuaderno tú mismo y si quieres puedes hacer algunos 
ejercicios. Empecemos importando TensorFlow. 

Voy a obtener los datos de Fashion MNIST usando tf.keras.datasets. Llamando al método load data, Obtengo los datos de entrenamiento y las etiquetas así como los datos de prueba y las etiquetas.

```python
# Load the Fashion MNIST dataset
fmnist = tf.keras.datasets.fashion_mnist
# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
```

![img_17.png](ims%2FW2%2Fimg_17.png)

Para más detalles sobre esto, vuelve a revisar el anterior vídeo. Los datos de una imagen en particular es una cuadrícula de valores desde cero hasta 255 con valores de los píxeles en escala de grises. Usando matplotlib, puedo dibujarlos como una imagen para facilitar la inspección. También puedo imprimir los valores en bruto para que podamos ver qué aspecto tienen. Puedes ver los valores en bruto de los números de píxel de cero a 255, y puedes observar la imagen real. 

![img_18.png](ims%2FW2%2Fimg_18.png)

Ahora diseñamos nuestro modelo. Como explicamos anteriormente, hay una capa de entrada con la forma de los datos y una capa de salida con la forma de las clases, y una capa oculta que trata de hallar las reglas entre ellas. Compilamos el modelo para encontrar la función de pérdida y el optimizador, y el objetivo de estos es como antes, hacer una conjetura de cuál es la relación entre los datos de entrada y los datos de salida, medir lo bien o lo mal que lo hacen mediante la función de pérdida, usa el optimizador para generar una nueva conjetura y repite.

![img_19.png](ims%2FW2%2Fimg_19.png)

Podemos tratar de ajustar las imágenes de entrenamiento a las etiquetas de entrenamiento. Lo haremos solo durante cinco epochs para ser breves. Pasamos unos 25 segundos entrenándolo durante cinco epochs y terminamos con una pérdida de cerca de 0.29. Eso significa que es bastante preciso en conjeturar la relación entre las imágenes y las etiquetas. No es tan genial, pero ten en cuenta que se ha hecho en solo 25 segundos con una red neuronal muy básica, lo que no está tampoco mal.

![img_20.png](ims%2FW2%2Fimg_20.png)

Pero una mejor medida del rendimiento se puede ver probando los datos de prueba. Estas son imágenes que la red aún no ha visto. Esperarías que el rendimiento fuera peor, pero si es aún peor, tienes un problema. Como puedes ver es alrededor de 0.345 de pérdida, lo que significa que es un poco menos preciso en el conjunto de prueba. No es tampoco genial, pero sabemos que estamos haciendo algo bien. 

 Tu trabajo ahora consiste en mirar el cuaderno, intentar los ejercicios y ver si cambiando los parámetros en la red neuronal o cambiando las epochs, hay alguna forma de obtener unos 0.71 de pérdida de precisión en los datos de entrenamiento y 0.66 de precisión en los de prueba, pruébalo tú mismo.

### Get Hands-on with computer vision (Lab 1)
[<- Return to INDEX 4](#index-4)

Ahora que ya ha visto el cuaderno de ejercicios, es hora de que lo pruebe por sí mismo. Puede encontrarlo 
aquí [C1_W2_Lab_1_beyond_hello_world.ipynb](notebooks%2FW2%2FC1_W2_Lab_1_beyond_hello_world.ipynb)
. También hemos proporcionado una serie de ejercicios que puede probar en la parte inferior del libro de trabajo. Éstos le ayudarán a hurgar y experimentar con el código, y le ayudarán con el código que tendrá que escribir al final de la semana, ¡así que realmente merece la pena dedicarles algo de tiempo! Le recomiendo que dedique al menos 1 hora a jugar con este cuaderno de ejercicios. ¡Realmente valdrá la pena su tiempo!

Cuando haya terminado con eso, lo siguiente es explorar las devoluciones de llamada. Una de las cosas que puede hacer con eso es entrenar una red neuronal hasta que alcance un umbral que usted desee, y entonces dejar de entrenar. Lo verá en el siguiente vídeo.

### Using Callbacks to control training
[<- Return to INDEX 4](#index-4)

![img_21.png](ims%2FW2%2Fimg_21.png)

Una pregunta que me hacen a menudo en este punto los programadores en particular cuando experimentan con diferentes números de epochs es ¿Cómo puedo parar el entrenamiento cuando alcanzo el punto en el que deseo estar? ¿Por qué siempre debo programar directamente un cierto número de epochs?

La buena nueva es que el bucle de entrenamiento soporta retrollamadas. En cada epoch, se puede retrollamar a una función de código, tras haber comprobado las métricas. Si son lo que quieres, entonces puedes cancelar el entrenamiento en ese punto. Echemos un vistazo. 

![img_22.png](ims%2FW2%2Fimg_22.png)

Este es nuestro código para entrenar la red neuronal para reconocer las imágenes de moda. En particular, mantén un ojo en la función model.fit que ejecuta el bucle de entrenamiento. Puedes ver eso aquí. Lo que haremos ahora es escribir una retrollamada en Python. 

![img_23.png](ims%2FW2%2Fimg_23.png)

Este es el código. Está implementada como una clase separada, pero puede estar en línea con el otro código. No necesita estar en un archivo separado. En ella, implementaremos la función on_epoch_end function, que es llamada por la retrollamada cada vez que un epoch termina. También envía un objeto de registro que contiene mucha información importante sobre el estado actual del entrenamiento. 

Por ejemplo, la pérdida actual está disponible en los registros, podemos consultarlo hasta cierto valor. Por ejemplo, aquí estoy comprobando si la pérdida es menor que 0.4 para cancelar el propio entrenamiento.

![img_24.png](ims%2FW2%2Fimg_24.png)

Ahora que tenemos la retrollamada volvamos al resto del código, hay dos modificaciones que debemos hacer. Primero, instanciamos la clase que acabamos de crear, lo hacemos con este código. 

![img_25.png](ims%2FW2%2Fimg_25.png)

En model.fit, usé los parámetros de la retrollamada y paso esta instancia de la clase. Veamos esto en acción.

### See how to implement Callbacks (Lab 2)
[<- Return to INDEX 4](#index-4)

Experimente con el uso de devoluciones de llamada en este 
cuaderno [C1_W2_Lab_2_callbacks.ipynb](notebooks%2FW2%2FC1_W2_Lab_2_callbacks.ipynb)
 -- ¡trabaje a través de él para ver cómo funcionan! 

### Walk through a notebook with Callbacks
[<- Return to INDEX 4](#index-4)

Echemos un vistazo al código de las retrollamadas y veamos cómo funciona. Puedes ver el código aquí. Este es el cuaderno con el código ya escrito, y esta es la clase que definimos para el manejo de la retrollamada

```python
import tensorflow as tf

# Instantiate the dataset API
fmnist = tf.keras.datasets.fashion_mnist

# Load the dataset
(x_train, y_train),(x_test, y_test) = fmnist.load_data()

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0
```

y aquí es donde instanciamos la clase callback misma. 

```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Halts the training when the loss falls below 0.4

    Args:
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
    '''

    # Check the loss
    if(logs.get('loss') < 0.4):

      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True

# Instantiate class
callbacks = myCallback()
```

Finalmente, aquí es donde configuramos la retrollamada como parte del bucle de entrenamiento. Cuando empezamos a entrenar, 
note que hemos pedido entrenar durante 10 epochs.

```python
# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with a callback
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```

Vigila las pérdidas que entrena. Deseamos que termine cuando sea más baja que 0.4 al final del primer epoch ya nos estamos acercando. Al iniciar el segundo epoch ya ha caído por debajo de 0.4, 
pero la retrollamada no ha sido alcanzada aún. Esto se debe a que se ha configurado para que sea al final del epoch. 

```commandline
Epoch 1/10
1875/1875 [==============================] - 9s 4ms/step - loss: 0.4734 - accuracy: 0.8315
Epoch 2/10
1867/1875 [============================>.] - ETA: 0s - loss: 0.3607 - accuracy: 0.8672
Loss is lower than 0.4 so cancelling training!
1875/1875 [==============================] - 7s 4ms/step - loss: 0.3609 - accuracy: 0.8671
```

Hacer esto es una buena práctica, porque con algunos datos y algunos algoritmos, la pérdida puede variar arriba y abajo durante el epoch, 
porque todos los datos aún no han sido procesados. Prefiero esperar hasta el final para estar seguro. Ahora la época ha terminado, 
la pérdida es 0.3563, podemos ver que el entrenamiento termina, a pesar que solo hemos hecho dos epochs. 

Ten en cuenta que estamos seguros de que pedimos cinco epochs y que hemos terminado después de dos, porque la pérdida es menor que 0.4, 
que es lo que comprobamos en la retrollamada. Está muy bien.

### Week 2 Quiz
[<- Return to INDEX 4](#index-4)

1. What is the resolution of the 70,000 images from the Fashion MNIST dataset?

   - [ ] 100x100 Color
   - [ ] 28x28 Color
   - [ ] 82x82 Greyscale
   - [X] 28x28 Greyscale

   > Correct
   > Spot on!

2. Why are there 10 output neurons in the Neural Network used as an example for the Computer Vision Problem?

   - [ ] Purely arbitrary
   - [ ] To make it train 10x faster
   - [X] There are 10 different labels
   - [ ] To make it classify 10x faster

   > Correct
   > Exactly! There are 10 output neurons because we have 10 classes of clothing in the dataset. These should always match.

3. What does Relu do?

   - [ ] It only returns x if x is less than zero
   - [ ] For a value x, it returns 1/x
   - [ ] It returns the negative of x
   - [X] It only returns x if x is greater than zero

   > Correct
   > Correct! The rectifier or ReLU (Rectified Linear Unit) activation function returns x if x is greater than zero.

4. Why do you split data into training and test sets?

   - [ ] To train a network with previously unseen data
   - [ ] To make training quicker
   - [X] To test a network with previously unseen data
   - [ ] To make testing quicker

   > Correct
   > Nailed it! Splitting the data into training and test sets allows you to test the network with unseen data.

5. True or False: The on_epoch_end function sends a logs object with lots of great information about the current state of training at the start of every epoch.

   - [ ] True
   - [X] False

   > Correct
   > Absolutely! The function activates at the end of every epoch.

6. Why do you set the callbacks= parameter in your fit function?

   - [ ] So that the training loops performs all epochs
   - [ ] Because it accelerates the training
   - [X] So, on every epoch you can call back to a code function

   > Correct
   > That’s right! You can have it check the metrics and stop the training.

### Lecture Notes Week 2
[<- Return to INDEX 4](#index-4)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C1_W2.pdf](notes%2FC1_W2.pdf)
.

#### Derechos de autor
Estas diapositivas se distribuyen bajo licencia Creative Commons.

[DeepLearning.](https://www.deeplearning.ai/)
AI pone estas diapositivas a su disposición con fines educativos. No puede utilizar ni distribuir estas diapositivas con fines comerciales. Puede hacer copias de estas diapositivas y utilizarlas o distribuirlas con fines educativos siempre que cite a
[DeepLearning](https://www.deeplearning.ai/)
.AI como fuente de las mismas.

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