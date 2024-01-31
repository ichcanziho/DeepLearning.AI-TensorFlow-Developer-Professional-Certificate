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

![img.png](ims%2FW1%2Fimg.png)

En este curso, los estudiantes aprenderán a utilizar TensorFlow para el procesamiento de texto en el procesamiento de lenguaje natural. El instructor señala que el texto es más complejo que las imágenes, ya que puede contener frases largas o cortas, y se debe decidir si se procesan caracteres individuales o palabras. Se enfocarán en cómo convertir el texto en números comprensibles para las redes neuronales, abordando preguntas como "¿cómo representar la palabra 'gato' como números?" y "¿cómo manejar diferentes longitudes de frases?"

![img_1.png](ims%2FW1%2Fimg_1.png)

También se tratará la cuestión de los tokens de vocabulario. En la primera semana, se enseñará a cargar, preprocesar y configurar datos de texto para alimentarlos a una red neuronal. El instructor muestra entusiasmo por el curso y anima a los estudiantes a continuar al siguiente video para comenzar.

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


### Introduction
[<- Return to INDEX 1](#index-1)

![img_2.png](ims%2FW1%2Fimg_2.png)

En los cursos anteriores de esta especialización, obtuviste una introducción al aprendizaje automático y al aprendizaje profundo con un gran enfoque en los problemas de visión por ordenador. Aprendió acerca de las redes neuronales y cómo pueden igualar patrones para realizar clasificaciones. Y luego cómo puedes darles nuevos datos y hacer que predigan lo que podrían estar viendo. Aprendió a hacer que esto sea un poco más inteligente para las imágenes usando circunvoluciones para identificar las características de las imágenes y clasificarlas en función de esas en lugar de simplemente coincidir con píxeles sin formato. Esto le ayudó a clasificar las imágenes para más estilos del mundo real en lugar de utilizar un entorno muy controlado.

![img_3.png](ims%2FW1%2Fimg_3.png)

En este curso vamos a volver a construir modelos, pero nos centraremos en el texto y cómo construir el clasificador se basa en modelos de texto. Empezaremos por ver el sentimiento en el texto y aprenderemos a crear modelos que entiendan el texto que está entrenado en el texto etiquetado y, a continuación, podemos clasificar el texto nuevo en función de lo que hayan visto.

Cuando tratábamos con imágenes, era relativamente fácil para nosotros alimentarlas en una red neuronal, ya que los valores de píxel ya eran números. Y la red podría aprender parámetros de funciones que podrían usarse para ajustar clases a etiquetas. Pero, ¿qué pasa con el texto? ¿ Cómo podemos hacer eso con frases y palabras?

### Word based encodings
[<- Return to INDEX 1](#index-1)

![img_4.png](ims%2FW1%2Fimg_4.png)

Podríamos tomar codificaciones de caracteres para cada carácter de un conjunto. Por ejemplo, los valores ASCII. ¿ Pero eso nos ayudará a entender el significado de una palabra?

![img_5.png](ims%2FW1%2Fimg_5.png)

Así que, por ejemplo, considere la palabra 'ESCUCHAR' como se muestra aquí. Una codificación de caracteres simple común es ASCII, el código estándar americano para el intercambio de información con los valores como se muestra aquí. Así que podría pensar que podría tener una palabra como LISTEN codificada usando estos valores.

![img_6.png](ims%2FW1%2Fimg_6.png)

Pero el problema con esto, por supuesto, es que la semántica de la palabra no está codificada en las letras. Esto podría demostrarse usando la palabra «SILENT », que tiene un significado muy diferente y casi opuesto, pero con exactamente las mismas letras. Así que parece que entrenar una red neuronal con sólo las letras podría ser una tarea desalentadora. 

![img_7.png](ims%2FW1%2Fimg_7.png)

Entonces, ¿qué tal si tenemos en cuenta las palabras? ¿ Qué pasaría si pudiéramos dar un valor a las palabras y utilizar esos valores en la formación de una red? Ahora podríamos estar llegando a alguna parte. Así que, por ejemplo, considere esta frase, Amo a mi perro. 

¿ Qué tal darle un valor a cada palabra? Lo que sea ese valor no importa. Es solo que tenemos un valor por palabra, y el valor es el mismo para la misma palabra cada vez. Entonces, una codificación simple para la oración sería, por ejemplo, dar a la palabra 'I' el valor uno. A continuación, podríamos dar a las palabras «Amor », «mi» y «perro» los valores 2 , 3 y 4 respectivamente. 
Así que entonces la frase, Amo a mi perro sería codificado como 1, 2, 3, 4

![img_8.png](ims%2FW1%2Fimg_8.png)

Así que ahora, ¿y si tengo la frase, amo a mi gato? Bueno, ya hemos codificado las palabras 'Amo mi' como 1, 2, 3. Así que podemos reutilizarlos, y podemos crear un nuevo token para gato, que no hemos visto antes. Así que hagamos que sea el número 5. 

![img_9.png](ims%2FW1%2Fimg_9.png)

Ahora, si solo miramos los dos conjuntos de codificaciones, podemos comenzar a ver alguna similitud entre las oraciones. Me encanta mi perro es 1, 2, 3, 4 y amo a mi gato es 1, 2, 3, 5. Así que esto es al menos un comienzo y cómo podemos empezar a entrenar una red neuronal basada en palabras. Afortunadamente, TensorFlow y Care Ask nos dan algunas API que hacen que sea muy sencillo hacer esto. Los veremos a continuación.

### Using APIs
[<- Return to INDEX 1](#index-1)

![img_10.png](ims%2FW1%2Fimg_10.png)

Aquí está el código para codificar las dos frases de las que acabamos de hablar. Vamos a desempaquetarlo línea por línea. Tensorflow y keras nos dan varias maneras de codificar palabras, pero en la que me voy a centrar es el tokenizador. Esto manejará el trabajo pesado para nosotros, generando el diccionario de codificaciones de palabras y creando vectores fuera de las oraciones. 

![img_11.png](ims%2FW1%2Fimg_11.png)

- Pondré las oraciones en una matriz. Tenga en cuenta que ya he mayúsculas 'I' tal como está al principio de la oración. Luego creo una instancia del tokenizer. 
- Un parámetro pasivo num se dirige a él. En este caso, estoy usando 100, que es demasiado grande, ya que solo hay cinco palabras distintas en estos datos. Si estás creando un conjunto de entrenamiento basado en un montón de texto, normalmente no sabes cuántas palabras únicas hay en ese texto. Entonces, al establecer este hiperparámetro, lo que hará el tokenizador es tomar las 100 palabras principales por volumen y simplemente codificarlas. 
- El método de ajuste en los textos del tokenizador toma los datos y los codifica.
- El tokenizer proporciona una propiedad de índice de palabras que devuelve un diccionario que contiene pares de valores clave, donde la clave es la palabra, y el valor es el token de esa palabra, que puede inspeccionar simplemente imprimiéndola.

![img_12.png](ims%2FW1%2Fimg_12.png)

Puedes ver los resultados aquí. Recuerde cuando dijimos que la palabra que estaba en mayúsculas, tenga en cuenta que está en minúsculas aquí. 

![img_13.png](ims%2FW1%2Fimg_13.png)

Eso es otra cosa que el tokenizador hace por ti. Elimina la puntuación. Esto es realmente útil si considera este caso. Aquí, he añadido otra frase, «¡Amas a mi perro!» pero hay algo muy diferente en ello. He añadido una exclamación después de la palabra «¡perro!» Ahora, ¿debería tratarse esto como una palabra diferente a sólo perro? Bueno, por supuesto que no. Así que los resultados del código que vimos anteriormente con este nuevo corpus de datos, se verán así

![img_14.png](ims%2FW1%2Fimg_14.png)

Tenga en cuenta que todavía sólo tenemos 'perro' como llave. Que la exclamación no afectó a esto, y por supuesto, tenemos una nueva clave para la palabra 'usted' que fue detectada. Así que ha visto los comienzos del manejo de textos creando codificaciones basadas en palabras de ese texto, con un flujo intensivo de código muy simple y keras. En el siguiente video, vamos a echar un vistazo al código y ver cómo funciona.

### Have questions, issues or ideas?
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


### Check out the code - Week 1 - (Lab 1)
[<- Return to INDEX 1](#index-1)

En el próximo vídeo, Laurence hará un screencast del primer 
cuaderno de [C3_W1_Lab_1_tokenize_basic.ipynb](notebooks%2FW1%2FC3_W1_Lab_1_tokenize_basic.ipynb)
 esta semana. Puede tenerlo abierto para poder seguirlo mientras él lo recorre.

### Notebook for lesson 1 - Week 1
[<- Return to INDEX 1](#index-1)

Aquí puedes ver el código que vimos en la lección. En primer lugar, utilizarás la clase tokenizer y esta clase tokenizadora se encuentra en tensorflow.keras.preprocessing.text.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
```

La clase tokenizadora se encargará de gestionar los tokens, convertir las frases en flujos de fichas y todo ese tipo de cosas. Así que echémosle un vistazo para empezar. Así que aquí tengo mi lista de oraciones que acabo de leer, amo a mi perro y amo a mi gato, tenga en cuenta que la I está en mayúscula. 

```python
# Define input sentences
sentences = [
    'i love my dog',
    'I, love my cat'
    ]
```

Y lo que va a hacer el tokenizador es que, cuando cree una instancia del mismo, le pase el número de palabras que quiero tener como máximo número de entradas en el diccionario. Así que en este caso solo hay 5 palabras diferentes, como puedes ver aquí, amo a mi perro y luego a mi gato. Así que adormecer las palabras por ser mayores de 5 es algo redundante. 

Pero como utilizas corpus de textos más grandes, si tienes miles de oraciones que quieres clasificar, es difícil que acabes intentando averiguar el número único de palabras que hay en todas ellas. Entonces, lo que puedes hacer es pasar este parámetro y decir: «Dime las 100 palabras más comunes de todo el corpus», dime las 1000 palabras más comunes o lo que sea. Así que lo estoy configurando por defecto en 100, aunque realmente no necesito más de 5.

```python
# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words = 100)
```

Luego, cuando utilice el método de ajuste de textos del tokenizador y le pase esta lista, lo que va a hacer es una serie de cosas. Lo primero que veremos es que va a crear ese índice de palabras para nosotros, donde el índice de palabras es una lista de pares de valores clave. 

```python
# Generate indices for each word in the corpus
tokenizer.fit_on_texts(sentences)
```

Donde la clave es la palabra y el valor es el símbolo de esa palabra y podemos imprimirlo aquí. 

```python
# Get the indices and print it
word_index = tokenizer.word_index
print(word_index)
```

```commandline
{'i': 1, 'love': 2, 'my': 3, 'dog': 4, 'cat': 5}
```

Este es el primer paso que tendrá que dar para preparar los datos basados en texto para entrenar redes neuronales. Esto se usará más adelante con algo llamado incrustación. Pero antes de hacer eso, solo queremos convertir nuestras oraciones en listas basadas en fichas y hacer que todas esas listas tengan el mismo tamaño

### Text to sequence
[<- Return to INDEX 1](#index-1)

![img_15.png](ims%2FW1%2Fimg_15.png)

En el video anterior, usted vio cómo tokenizar las palabras y oraciones, construyendo un diccionario de todas las palabras para hacer un corpus. El siguiente paso será convertir sus oraciones en listas de valores basadas en estos tokens. Una vez que los tengas, es probable que también necesites manipular estas listas, no menos importante para hacer que cada frase tenga la misma longitud, de lo contrario, puede ser difícil entrenar una red neuronal con ellas.

![img_16.png](ims%2FW1%2Fimg_16.png)

Recuerde que cuando estábamos haciendo imágenes, definimos una capa de entrada con el tamaño de la imagen que estamos alimentando en la red neuronal. En los casos en que las imágenes tenían un tamaño diferente, las cambiaríamos para que se ajustaran. Bueno, vas a enfrentarte a lo mismo con el texto. Afortunadamente, TensorFlow incluye API para manejar estos problemas. Los veremos en este video. 

Comencemos con la creación de una lista de secuencias, las oraciones codificadas con los tokens que generamos y he actualizado el código en el que hemos estado trabajando a esto. En primer lugar, he añadido otra frase al final de la lista de oraciones. Tenga en cuenta que todas las oraciones anteriores tenían cuatro palabras en ellas. Así que este es un poco más largo. Usaremos eso para demostrar el relleno en un momento. 

![img_17.png](ims%2FW1%2Fimg_17.png)

La siguiente pieza de código es este, donde simplemente llamo al tokenizador para obtener textos a secuencias, y los convertirá en un conjunto de secuencias para mí. Entonces, si ejecuto este código, este será el resultado. 

![img_18.png](ims%2FW1%2Fimg_18.png)

En la parte superior está el nuevo diccionario. Con nuevos tokens para mis nuevas palabras como increíble, pensar, es, y hacer. En la parte inferior está mi lista de oraciones que han sido codificadas en listas enteras, con los tokens reemplazando las palabras. 

![img_19.png](ims%2FW1%2Fimg_19.png)

Así, por ejemplo, amo a mi perro se convierte en 4, 2, 1, 3. Una cosa muy útil sobre esto que usará más adelante es el hecho de que el texto a las secuencias llamadas puede tomar cualquier conjunto de oraciones, por lo que puede codificarlas en función del conjunto de palabras que aprendió del que se pasó en forma en los textos. 

![img_20.png](ims%2FW1%2Fimg_20.png)

Esto es muy significativo si piensas un poco más adelante. Si entrenas una red neuronal en un corpus de textos, y el texto tiene un índice de palabras generado a partir de ella, entonces cuando quieras hacer inferencia con el modelo de tren, tendrás que codificar el texto que quieres inferir con el mismo índice de palabras, de lo contrario no tendría sentido. 

![img_21.png](ims%2FW1%2Fimg_21.png)

Entonces, si considera este código, ¿cuál espera que sea el resultado? Hay algunas palabras familiares aquí, como el amor, mi, y el perro, pero también algunas anteriormente invisibles. 

![img_22.png](ims%2FW1%2Fimg_22.png)

Si ejecuto este código, esto es lo que obtendría. He añadido el diccionario debajo para mayor comodidad. Así que realmente amo a mi perro todavía estaría codificado como 4, 2, 1, 3, que es 'amo a mi perro' con 'realmente' perderse ya que la palabra no está en el índice de palabras,

![img_23.png](ims%2FW1%2Fimg_23.png)

 'mi perro ama a mi manatí' se codificaría en 1, 3, 1, que es sólo 'mi perro mi'.

### Looking more at the Tokenizer
[<- Return to INDEX 1](#index-1)

![img_24.png](ims%2FW1%2Fimg_24.png)

Entonces, ¿qué aprendemos de esto? En primer lugar, realmente necesitamos una gran cantidad de datos de entrenamiento para obtener un vocabulario amplio o podríamos terminar con frases como, mi perro, como acabamos de hacer. En segundo lugar, en muchos casos, es una buena idea, en lugar de ignorar palabras invisibles, poner un valor especial cuando se encuentra una palabra invisible. Puede hacer esto con una propiedad en el tokenizer. Echemos un vistazo.

![img_25.png](ims%2FW1%2Fimg_25.png)

Aquí está el código completo que muestra tanto las oraciones originales como los datos de prueba. Lo que he cambiado es agregar un token oov de propiedad al constructor tokenizer. Ahora puede ver que he especificado que quiero que el token oov para el vocabulario externo se use para palabras que no están en el índice de palabras. Puedes usar lo que quieras aquí, pero recuerda que debe ser algo único y distinto que no se confunda con una palabra real. 

![img_26.png](ims%2FW1%2Fimg_26.png)

Así que ahora, si ejecuto este código, conseguiré que mis secuencias de prueba se vean así. Pegé el índice de palabras debajo para que puedas buscarlo. La primera frase será, yo fuera de vocab, amo a mi perro. El segundo será, mi perro oov, mi oov Todavía no sintácticamente grande, pero lo está haciendo mejor. 

![img_27.png](ims%2FW1%2Fimg_27.png)

A medida que el corpus crece y más palabras están en el índice, es de esperar que las oraciones nunca vistas tengan una mejor cobertura. El siguiente es el relleno. Como mencionamos antes cuando estábamos construyendo redes neuronales para manejar imágenes. Cuando los alimentamos en la red para el entrenamiento, necesitábamos que fueran uniformes en tamaño.

A menudo, usamos los generadores para cambiar el tamaño de la imagen para que se ajuste, por ejemplo. Con los textos te enfrentarás a un requisito similar antes de poder entrenar con textos, necesitábamos tener algún nivel de uniformidad de tamaño, por lo que el relleno es tu amigo allí.

### Padding
[<- Return to INDEX 1](#index-1)

![img_30.png](ims%2FW1%2Fimg_30.png)

Así que he hecho algunos cambios en el código para manejar el relleno. Aquí está el listado completo y lo desglosaremos pieza por pieza. Primero, para usar las funciones de relleno, tendrá que importar secuencias de pad desde tensorflow.carastoppreprocessing.sequence.

Luego, una vez que el tokenizer ha creado las secuencias, estas secuencias se pueden pasar a secuencias de pad para tenerlas acolchadas así. El resultado es bastante sencillo. 

![img_31.png](ims%2FW1%2Fimg_31.png)

Ahora puede ver que la lista de oraciones se ha rellenado en una matriz y que cada fila de la matriz tiene la misma longitud. Esto se logró colocando el número apropiado de ceros antes de la oración. 

![img_32.png](ims%2FW1%2Fimg_32.png)

Entonces, en el caso de la oración 5.3.2.4, en realidad no hizo ninguna. 

![img_33.png](ims%2FW1%2Fimg_33.png)

En el caso de la oración más larga aquí, no tenía que hacer ninguna. A menudo verá ejemplos en los que el relleno está después de la oración y no antes como acabas de ver. 

![img_34.png](ims%2FW1%2Fimg_34.png)

Si usted, como yo, está más cómodo con eso, puede cambiar el código a esto, agregando el relleno de parámetros es igual a publicación. Es posible que haya notado que el ancho de la matriz era el mismo que la frase más larga.

Pero puede anular eso con el parámetro maxlen. Así, por ejemplo, si solo quieres que tus oraciones tengan un máximo de cinco palabras. Se puede decir que maxlen equivale a cinco como este. Esto, por supuesto, llevará a la pregunta. Si tengo oraciones más largas que la longitud máxima, entonces perderé información, pero de dónde. 

![img_35.png](ims%2FW1%2Fimg_35.png)

Al igual que con el relleno, el valor predeterminado es pre, lo que significa que perderás desde el principio de la oración. Si desea anular esto para que pierda desde el final en su lugar, puede hacerlo con el parámetro de truncamiento como este.

Así que ahora ha visto cómo codificar sus oraciones, cómo rellenarlas y cómo usar la indexación de Word para codificar oraciones previamente invisibles usando caracteres fuera de vocab. Pero lo has hecho con datos codificados muy simples. Echemos un vistazo a la acción codificada en un screencast y luego volveremos a ver cómo usar datos mucho más complejos.

### Check out the code - Week 1 - (Lab 2)
[<- Return to INDEX 1](#index-1)

Aquí tiene el segundo 
cuaderno de [C3_W1_Lab_2_sequences_basic.ipynb](notebooks%2FW1%2FC3_W1_Lab_2_sequences_basic.ipynb)
 esta semana y Laurence hablará de él en el próximo screencast.

### Notebook for lesson 2 - Week 1
[<- Return to INDEX 1](#index-1)

Echemos un vistazo a algunos de los procesos básicos de preprocesamiento de texto que hemos estado realizando. En primer lugar, vamos a usar un par de API en TensorFlow. Está el tokenizador y las secuencias de botones.

El tokenizador se encuentra en tensorflow.keras.preprocessing.text y las secuencias pad se encuentran en tensorflow.keras.preprocessing.sequence. 

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

Nuestro corpus de trabajo inicial en el que basaremos nuestra tokenización se detalla aquí. Esta es mi lista de oraciones. Amo a mi perro, amo a mi gato. Amas a mi perro, ¿y crees que mi perro es increíble? 

```python
# Define your input texts
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
```

En el tokenizador, voy a crear una instancia de un nuevo tokenizador. Voy a decir que el número máximo de palabras en su corpus debe ser 100. Obviamente tenemos menos de 100 palabras aquí o 100 palabras diferentes, únicas y distintas aquí. También voy a especificarlas fuera del vocabulario. El token debería tener este aspecto. Solo voy a decir que es OOV. Puedo poner lo que quiera aquí. Pero siempre debe ser algo que no esperes ver en el cuerpo del texto, por eso pongo corchetes angulares y OOV de esta manera.

```python
# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
```

A continuación, cuando diga frases de tokenizer.fit_on_text, lo que hará será generar un tokenizador mediante la creación de un conjunto único de valores clave para las palabras únicas que están aquí. 

```python
# Tokenize the input sentences
tokenizer.fit_on_texts(sentences)
```

Me convertiré en un amor clave o me convertiré en una clave, ese tipo de cosas. Creará un índice de palabras donde haya pares de valores clave, donde una palabra sea la clave y esa clave tenga un valor. Por ejemplo, el amor se convierte en una clave, yo se convierte en una clave, el perro se convierte en una clave. Ese tipo de cosas. Ese índice de palabras lo puedo extraer del tokenizador simplemente llamando a su propiedad de índice de palabras. 

```python
# Get the word index dictionary
word_index = tokenizer.word_index
```

Puedo obtener una variable de índice de palabras que podamos ver. Lo siguiente es poder usar secuencias en el tokenizador. 



```python
# Generate list of token sequences
sequences = tokenizer.texts_to_sequences(sentences)

# Print the result
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
```

Luego convertirá las oraciones de texto que tiene en secuencias, reemplazando cada palabra de la oración con el valor de la clave para esa palabra en 
particular. Por ejemplo, si la palabra yo puede tener la clave 2 y el amor tiene la clave 3. 

En vez de decir «me encanta», tendría 2, 3 en una secuencia, ese tipo de cosas. Eso me dará las secuencias. 

```commandline
Word Index =  {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

Sequences =  [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
```

A continuación, utilizaré secuencias de almohadillas para hacer que todas estas secuencias tengan el mismo tamaño o el mismo ancho. 
Verás, algunas de estas oraciones tienen cuatro palabras. Algunas tienen más, creo que esta tiene siete palabras. 
Cuando entrenamos una red neuronal, queremos que todos los datos tengan el mismo tamaño. 


```python
# Pad the sequences to a uniform length
padded = pad_sequences(sequences, maxlen=5)

# Print the result
print("\nPadded Sequences:")
print(padded)
```

 En este caso, si el maxlen es cinco, cogeremos las cinco primeras palabras. ¿Crees que es mi perro y perderemos las dos últimas palabras? Si no tuviésemos la opción de truncar publicaciones iguales, el valor predeterminado sería pre o si lo configuramos así, entonces se usarán las últimas cinco palabras, es decir, piensa que mi perro es increíble, ese tipo de cosas.

```commandline
Padded Sequences:
[[ 0  5  3  2  4]
 [ 0  5  3  2  7]
 [ 0  6  3  2  4]
 [ 9  2  4 10 11]]
```

Cuando empezamos a ver frases que son diferentes a estas, por ejemplo, me encanta mucho mi perro. En vez de decir «amo a mi perro», nos daremos cuenta de que vamos a utilizar un carácter fuera de vocabulario para esto, porque este no lo hemos aprendido. Esto se vuelve muy importante más adelante , cuando entrenes redes neuronales con estos datos. Porque si lo vas a entrenar con los símbolos que hay aquí, va a ser más difícil clasificar frases como estas porque hay palabras que el tokenizador no ha visto antes y, por lo tanto, la red neuronal no se va a entrenar con ellas

```python
# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
```

Una vez más, es un caso clásico en el que cuantos más datos de entrenamiento tengas, mejor. Pero veamos cómo se comportará el tokenizador en este escenario de todos modos. En este caso, voy a crear una secuencia de prueba a partir de los datos de prueba en la que volveré a usar el tokenizador y pasaré los datos de prueba para generar algunas secuencias. Voy a imprimirlas y luego las rellenaré. 

```python
# Generate the sequences
test_seq = tokenizer.texts_to_sequences(test_data)

# Print the word index dictionary
print("\nWord Index = " , word_index)

# Print the sequences with OOV
print("\nTest Sequence = ", test_seq)

# Print the padded result
padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)
```

De nuevo, puede ver que los estoy rellenando, no estoy especificando ningún comportamiento aquí. La longitud máxima será la longitud de la oración más larga. El relleno será previo y el truncamiento será previo. Si lo ejecuto y les echamos un vistazo. Ahora podemos ver que las secuencias de mis pruebas se vuelven bastante interesantes porque dije que realmente quería a mi perro. Mi secuencia de prueba es 5, 1, 3, 2, 4. 

Cinco soy yo, uno, por supuesto, es el símbolo de falta de vocabulario, y luego 3,2,4 son love my dog. En blanco, amo a mi perro es lo que realmente se ve en la secuencia de la prueba. Esta frase dice: «Mi perro ama a mi manatí». La palabra ama es diferente de la palabra amor, así que no hay ningún símbolo para ello y, por supuesto, no hay ningún símbolo para el manatí. Pero ya hemos visto a mi perro y a mí antes. Así que terminamos con 2,4,1,2,1.

Dos, por supuesto, son míos, cuatro son perros, uno está fuera de vocabulario, dos es mío y uno está fuera de vocabulario. Hago que mi perro quede en blanco, mi espacio en blanco. Luego, cuando rellene las secuencias con este código, las 5,3,1,4 permanecerán iguales. Las frases 2,4,1,2,1 permanecerán iguales porque las frases tienen la misma longitud, por lo que no es necesario rellenarlas y truncarlas.

```commandline
Word Index =  {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}

Test Sequence =  [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]

Padded Test Sequence: 
[[0 0 0 0 0 5 1 3 2 4]
 [0 0 0 0 0 2 4 1 2 1]]
```

Por supuesto, si hubiera dicho aquí algo como maxlen igual a 10, y luego lo ejecutara. Tendremos mucho relleno y, dado que la experiencia de relleno predeterminada es previa, se rellenará de antemano. Esta es una visión básica de cómo usar el tokenizador y cómo usar las secuencias de teclado para preprocesar el texto. En el siguiente vídeo, analizaremos cómo cargar conjuntos de datos mucho más grandes en lugar de estos simples conjuntos de datos codificados de forma rígida, y veremos qué tipo de tokenizador y relleno haremos con ellos


### Sarcasm, really?
[<- Return to INDEX 1](#index-1)

![img_36.png](ims%2FW1%2Fimg_36.png)

En lo que va de la semana, has estado mirando los textos, y cómo tokenizar el texto, y luego convertir oraciones en secuencias usando las herramientas disponibles en TensorFlow. Lo hiciste usando algunas frases codificadas muy simples. Pero, por supuesto, cuando se trata de hacer problemas del mundo real, estarás usando muchos más datos que solo estas simples frases. Así que en esta lección, vamos a echar un vistazo a algunos conjuntos de datos públicos y cómo puede procesarlos para prepararlos para entrenar una red neuronal.

![img_37.png](ims%2FW1%2Fimg_37.png)

Comenzaremos con esta publicada por Rishabh Misra con detalles sobre Kaggle en este enlace. Es un conjunto de datos de dominio público CC0 realmente divertido en torno a la detección de sarcasmo. ¿ En serio? Sí, en serio. 

![img_38.png](ims%2FW1%2Fimg_38.png)

Este conjunto de datos es muy sencillo y sencillo, por no mencionar muy fácil de trabajar. Tiene tres elementos en él. La primera es sarcástica, es nuestra etiqueta. Es uno si el disco se considera sarcástico de lo contrario es cero. El segundo es un titular, que es sólo texto sin formato y el tercero es el enlace al artículo que describe el titular. Analizar el contenido de HTML, eliminar scripts, estilos, etc., está un poco más allá del alcance de este curso. Así que nos vamos a centrar en los titulares. 

![img_39.png](ims%2FW1%2Fimg_39.png)

Si descargas los datos de ese sitio de Kaggle, verás algo como esto. Como puede ver, es un conjunto de entradas de lista con pares nombre-valor donde el nombre es enlace del artículo, título e is_sarcástico y los valores son como se muestra. 

![img_40.png](ims%2FW1%2Fimg_40.png)

Para que sea mucho más fácil cargar estos datos en Python, hice un pequeño ajuste a los datos para que se vean así, lo cual puede sentirse libre de hacer o puede descargar mi conjunto de datos modificado desde el enlace en el co-laboratorio para esta parte del curso. Una vez que tenga los datos como estos, entonces es realmente fácil cargarlos en Python. 

![img_41.png](ims%2FW1%2Fimg_41.png)

Echemos un vistazo al código. Entonces primero debes importar JSON. Esto le permite cargar datos en formato JSON y crear automáticamente una estructura de datos de Python a partir de ella. Para ello, simplemente abra el archivo y páselo a json.load y obtendrá una lista con listas de los tres tipos de datos: titulares , URL y etiquetas is_sarcásticas. 

![img_42.png](ims%2FW1%2Fimg_42.png)

Porque quiero que las oraciones como una lista propia pasen al tokenizer, puedo crear una lista de oraciones y más tarde, si quiero que las etiquetas para crear una red neuronal, también puedo crear una lista de ellas. Mientras estoy en ello, también puedo hacer URL aunque no voy a usarlas aquí, pero es posible que quieras. 

Ahora puedo iterar a través de la lista que se creó con un elemento for en el bucle del almacén de datos. Para cada elemento, puedo copiar el titular a mis oraciones, el is_sarcastic a mis etiquetas y el article_link a mis URL. Ahora tengo algo con lo que puedo trabajar en el tokenizador, así que veamos eso a continuación.

### Working with the Tokenizer
[<- Return to INDEX 1](#index-1)

![img_43.png](ims%2FW1%2Fimg_43.png)

Este código es muy similar a lo que viste en los videos anteriores, pero veamos línea por línea. Acabamos de crear frases menos de los titulares, en el conjunto de datos del sarcasmo. Entonces, al llamar a tokenizer.fit en textos, generará el índice de palabras e inicializaremos el tokenizer.

![img_44.png](ims%2FW1%2Fimg_44.png)

Podemos ver el índice de la palabra como antes llamando a la propiedad de índice de la palabra. Tenga en cuenta que esto devuelve todas las palabras que el tokenizer vio al tokenizar las oraciones. Si especifica palabras num para obtener el top 1000 o lo que sea, puede estar confundido al ver algo mayor que eso aquí. Es un error fácil de cometer, pero lo clave a recordar, es que cuando toma el top 1000 o lo que haya especificado, lo hace en el texto para secuenciar este proceso. Nuestro índice de palabras es mucho más grande que con el ejemplo anterior. Así que veremos una mayor variedad de palabras en él. 

![img_45.png](ims%2FW1%2Fimg_45.png)

Aquí hay unos pocos. Ahora crearemos las secuencias a partir del texto, así como rellenarlas. 

![img_46.png](ims%2FW1%2Fimg_46.png)

Aquí está el código para hacer eso. Es muy similar a lo que hiciste antes, y aquí está la salida. Primero, tomé el primer titular en el conjunto de datos y mostré su salida. Podemos ver que ha sido codificado con los valores de las claves que son la palabra correspondiente en la frase. 

![img_47.png](ims%2FW1%2Fimg_47.png)

Este es el tamaño de la matriz acolchada. Teníamos 26.709 frases, y estaban codificadas con relleno, para conseguir hasta 40 palabras de largo que era la longitud de la palabra más larga. Podrías truncar esto si quieres, pero lo mantendré en 40. Eso es todo para procesar el conjunto de datos del sarcasmo. Echemos un vistazo a eso en acción en un reparto de pantalla.

### News Headlines dataset for sarcasm detection
[<- Return to INDEX 1](#index-1)

A continuación se presenta el conjunto de datos de dominio público basado en el sarcasmo, tal y como se muestra en el vídeo anterior. 

El enlace se proporciona aquí para su comodidad:

[Sarcasmo en los titulares de las noticias Dataset by Rishabh Misra](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)

![img_28.png](ims%2FW1%2Fimg_28.png)

### Check out the code - Week 1 - (Lab 3)
[<- Return to INDEX 1](#index-1)

Este es el 
cuaderno [C3_W1_Lab_3_sarcasm.ipynb](notebooks%2FW1%2FC3_W1_Lab_3_sarcasm.ipynb)
 que se mostrará en el próximo screencast.

### Notebook for lesson 3 - Week 1
[<- Return to INDEX 1](#index-1)

Así que lo primero que voy a hacer es descargar el conjunto de datos modificado. Así que modifiqué el conjunto de datos de Kaggle, tal como lo mostré en el vídeo, para crear este sarcasm.json. Esto nos facilita un poco la carga en Python.

A continuación, importo JSON y, al importar JSON, tengo acceso a este objeto al que puedo llamar json.load y me facilita mucho la vida, ya que he descargado 
mi archivo y lo he almacenado en slash TMP slash sarcasm.json. Y luego puedo decir que si lo abro como F. 

```python
# Download the dataset
!wget https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json

import json

# Load the JSON file
with open("./sarcasm.json", 'r') as f:
    datastore = json.load(f)
```

Puedo crear esta lista llamada almacén de datos diciendo json dot load F. Y ahora se me han cargado todos los datos.

 Para la tokenización, quiero separar los titulares en dos frases y, más adelante, cuando usemos una red neuronal para clasificarlos, crearé etiquetas a partir de las puntuaciones del conjunto de datos

hora que tengo mis frases, veamos qué se necesita para convertirlas en un token y el código para ello ya debería resultarme muy familiar.
```python
# Initialize lists
sentences = []
labels = []
urls = []

# Append elements in the dictionaries into each list
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```

Voy a importar secuencias de pad de tensorflow keras antes de procesar la secuencia de puntos. Voy a especificar mi símbolo de falta de vocabulario y luego simplemente voy a incluir las oraciones en los textos. Así que ahora las frases son todo el corpus de titulares.

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Tokenizer class
tokenizer = Tokenizer(oov_token="<OOV>")

# Generate the word index dictionary
tokenizer.fit_on_texts(sentences)
```

Puedo echar un vistazo a mi índice de palabras e imprimiré la longitud de ese índice de palabras y aquí hay algo muy importante que podemos discutir. Y luego imprimiré el índice de palabras en sí

```python
# Print the length of the word index
word_index = tokenizer.word_index
print(f'number of words in word_index: {len(word_index)}')

# Print the word index
print(f'word_index: {word_index}')
print()
```

Finalmente, voy a crear un conjunto de secuencias para codificar las palabras de estas oraciones en los valores en los que la palabra es esa clave en particular. Y luego voy a rellenarlo. Voy a mantener esto por defecto, excepto por el hecho de que voy a rellenar el post. Así que en las oraciones más cortas quiero tener ceros al final y luego imprimirlas. Así que ejecutemos esto y veamos qué pasa.

```python
# Generate and pad the sequences
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# Print a sample headline
index = 2
print(f'sample headline: {sentences[index]}')
print(f'padded sequence: {padded[index]}')
print()

# Print dimensions of padded sequences
print(f'shape of padded sequences: {padded.shape}')
```
Así que podemos ver que ha descargado mi sarcasm.json. En este caso, está impreso 29657 con la longitud del índice de palabras que vemos aquí. Así que había 29657 palabras únicas en este corpus. Y luego imprimí mi palabra y, a continuación, podemos ver que mi vocabulario agotado era 12 es 2 oficina 3, hay 4, ese tipo de cosas. 

```commandline
number of words in word_index: 29657
word_index: {'<OOV>': 1, 'to': 2, 'of': 3, 'the': 4, 'in': 5 }
```
Y luego echo un vistazo a mis secuencias, generaré mi secuencia, generaré mi relleno y luego imprimo el relleno de la primera oración para que podamos verla. 
Podemos ver que la primera oración solo tenía unas pocas palabras y estaba rellenada con muchos ceros. Y luego veremos el tamaño de mi 
matriz acolchada que está saliendo. Así que son 26,709 oraciones que han sido codificadas en un porcentaje de 40 enteros. 

Eso me sugiere que la oración más larga del corpus tenía 40 palabras. Y, por ejemplo, esta en la que es un poco más corta. Está acolchado con ceros adicionales. Si hubiera querido truncarlo, podría haberlo hecho en las secuencias del pad. 
```commandline
sample headline: mom starting to fear son's web series closest thing she will have to grandchild
padded sequence: [  145   838     2   907  1749  2093   582  4719   221   143    39    46
     2 10736     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0]

shape of padded sequences: (26709, 40)
```

hora, una cosa importante es este índice de palabras. Así que muchas veces decimos que cuando estás creando la palabra cuando estás creando el tokenizador y 
lo colocas en el texto. Puedes especificar el número máximo de palabras que deseas. Este índice de palabras siempre será el mismo, independientemente 
del número que hayas establecido como máximo. 

Así, por ejemplo, si hubiera establecido un máximo de 100 palabras, el índice de palabras seguiría teniendo 29657. Solo tendría en cuenta las 100 cuando 
realmente esté creando las secuencias. Y luego tomará mis 100 palabras principales y las palabras que no estén entre esas 100 mejores, y 
las reemplazará con las palabras que no están en el vocabulario. 

Así que hay una cosa importante y no la voy a tratar aquí. Pero una cosa importante si vas a hacer esto de verdad más adelante es que verás que hay muchas palabras aquí, 
como ir y bajar y entrar y para las que van a estar entre tus 100 mejores. 

Así que si vas a truncar hasta el número más alto de palabras, tal vez quieras deshacerte de muchas palabras sin sentido como estas y luego centrarte en las palabras más importantes, las que tienen una semántica para ti. Y, de hecho, creo que la forma más sencilla de hacerlo sería eliminarlas de las propias oraciones antes de incluirlas en los textos. 


### Week 1 Quiz
[<- Return to INDEX 1](#index-1)

1. What is the name of the object used to tokenize sentences?

   - [X] Tokenizer
   - [ ] WordTokenizer
   - [ ] CharacterTokenizer
   - [ ] TextTokenizer

   > Correct
   > That's right! The Tokenizer object is used to tokenize the sentences.

2. What is the name of the method used to tokenize a list of sentences?

   - [ ] tokenize_on_text(sentences)
   - [ ] tokenize(sentences)
   - [X] fit_on_texts(sentences)
   - [ ] fit_to_text(sentences)

   > Correct
   > Correct! The method fit_on_texts is used for tokenizing a list of sentences.

3. Once you have the corpus tokenized, what’s the method used to encode a list of sentences to use those tokens?

   - [X] texts_to_sequences(sentences)
   - [ ] text_to_sequences(sentences)
   - [ ] texts_to_tokens(sentences)
   - [ ] text_to_tokens(sentences)

   > Correct
   > Correct! texts_to_sequences is the method used to encode a list of sentences with the tokens.

4. When initializing the tokenizer, how do you specify a token to use for unknown words?

   - [ ] out_of_vocab=<Token>
   - [X] oov_token=<Token>
   - [ ] unknown_word=<Token>
   - [ ] unknown_token=<Token>

   > Correct
   > That's right! oov_token is used to specify a token for unknown words.

5. If you don’t use a token for out of vocabulary words, what happens at encoding?

   - [ ] The word is replaced by the most common token
   - [X] The word isn’t encoded, and is skipped in the sequence
   - [ ] The word isn’t encoded, and the sequencing ends
   - [ ] The word isn’t encoded, and is replaced by a zero in the sequence

   > Correct
   > Correct! Without a specific out of vocabulary token, words that are not in the index are skipped.

6. If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?

   - [ ] Specify the input layer of the Neural Network to expect different sizes with dynamic_length
   - [X] Use the pad_sequences function from the tensorflow.keras.preprocessing.sequence namespace
   - [ ] Process them on the input layer of the Neural Network using the pad_sequences property
   - [ ] Make sure that they are all the same length using the pad_sequences method of the tokenizer

   > Correct
   > That's right! The pad_sequences function is used to make all the sequences the same length.

7. If you have a number of sequences of different length, and call pad_sequences on them, what’s the default result?

   - [ ] They’ll get cropped to the length of the shortest sequence
   - [X] They’ll get padded to the length of the longest sequence by adding zeros to the beginning of shorter ones
   - [ ] Nothing, they’ll remain unchanged
   - [ ] They’ll get padded to the length of the longest sequence by adding zeros to the end of shorter ones

   > Correct
   > Correct! By default, sequences are padded to the length of the longest sequence by adding zeros at the beginning.

8. When padding sequences, if you want the padding to be at the end of the sequence, how do you do it?

   - [ ] Call the padding method of the pad_sequences object, passing it ‘after’
   - [ ] Pass padding=’after’ to pad_sequences when initializing it
   - [X] Pass padding=’post’ to pad_sequences when initializing it
   - [ ] Call the padding method of the pad_sequences object, passing it ‘post’

   > Correct
   > That's right! The padding is added to the end of the sequences by passing padding='post' to pad_sequences.

### Week 1 Wrap up
[<- Return to INDEX 1](#index-1)

![img_29.png](ims%2FW1%2Fimg_29.png)

Esta semana, has visto cómo tokenizar palabras, convertirlas en números, y usar un diccionario para buscar qué palabra va con qué número. La próxima semana, lo llevarás al siguiente paso usando algo llamado Embeddings, que toma estos números y comienza a establecer sentimientos desde ellos, para que puedas empezar a clasificar y luego predecir los textos.

### Lecture Notes Week 1
[<- Return to INDEX 1](#index-1)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C3_W1.pdf](notes%2FC3_W1.pdf)
.

## Weekly Assignment - Explore the BBC News Archive
[<- Return to INDEX 0](#index-0)

### Assignment Troubleshooting Tips

He aquí algunas directrices generales antes de entregar sus tareas en este curso. Téngalas en cuenta no sólo para la tarea de esta semana, sino también para las siguientes:

1. Para las tareas que utilicen Coursera Labs en lugar de Google Colab: Asegúrese de guardar su trabajo antes de hacer clic en el botón Submit. De lo contrario, podría recibir un mensaje de error como el de la celda de código que aparece a continuación. Recuerde que todo lo que tiene que rellenar dentro de las funciones calificadas se inicializa en None.
    ```commandline
      Failed test case: x has incorrect type.
      Expected:
      some.Type,
      but got:
      <class 'NoneType'>.
    ```

2. Para las tareas que utilizan Coursera Labs en lugar de Google Colab: Por favor, no cambie el nombre del cuaderno. El calificador buscará el nombre de archivo original y sus metadatos asociados, por lo que debe trabajar en el archivo que se abre automáticamente al hacer clic en el botón Launch Notebook. Si intenta enviar un cuaderno renombrado, es probable que también obtenga un error como el que se muestra arriba.

3. Por favor, no modifique ningún código fuera de las etiquetasSTART CODE HERE yEND CODE HERE. Su solución sólo debe colocarse entre estos marcadores para garantizar una calificación correcta. Modificar los parámetros de las funciones y otras celdas de prueba probablemente romperá el calificador. Si desea experimentar con ellas, puede hacerlo después de haber superado con éxito la tarea.

4. Después de seguir los consejos anteriores y el calificador le sigue dando 0/100, es posible que los metadatos necesarios para la calificación estén dañados. Por favor, obtenga un nuevo cuaderno de laboratorio refrescando su espacio de trabajo
([instrucciones aquí](https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/4Um1d/optional-downloading-your-notebook-and-refreshing-your-workspace)
). A continuación, copie sus soluciones en el nuevo cuaderno. Asegúrese de que todas las celdas siguen funcionando como se espera y, a continuación, vuelva a enviarlas.

5. Si tiene más preguntas, por favor cree un tema en la comunidad Discourse en lugar de los foros de discusión de Coursera. Puede unirse 
[siguiendo las instrucciones aquí](https://www.coursera.org/learn/natural-language-processing-tensorflow/ungradedLti/jF6xb/important-have-questions-issues-or-ideas-join-our-community-on-discourse)
. Obtendrá ayuda allí más rápidamente porque varios mentores y sus compañeros de aprendizaje están supervisando los mensajes. Sólo asegúrese de crear el tema en la categoría correcta del curso.

### Downloading your Notebook and refreshing your workspace

La mayoría de las tareas de este curso utilizan Coursera Labs para proporcionar un entorno de cuaderno y calificar su trabajo. Puede haber algunos casos en los que desee descargar sus cuadernos o actualizar su espacio de trabajo para empezar desde cero. Este elemento de lectura describe los pasos para hacerlo.

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

### Explore the BBC news archive

Para este ejercicio obtendrá el 
archivo de texto de la BBC [C3W1_Assignment.ipynb](notebooks%2FW1%2FC3W1_Assignment.ipynb)
. Su trabajo consistirá en tokenizar el conjunto de datos, eliminando las stopwords comunes. 

> IMPORTANTE PARA UNA CALIFICACIÓN CORRECTA:- ¡No olvide guardar su cuaderno antes de enviarlo!- No borre las celdas, ya que incluyen metadatos importantes para la calificación- Rellene sus soluciones dentro de los espacios proporcionados. Puede añadir nuevas celdas pero éstas serán omitidas por el calificador.

Si tiene alguna duda sobre las tareas de este curso, solicite ayuda en nuestra comunidad. Si aún no lo ha hecho
[por favor, ¡haga clic aquí y siga las instrucciones para poder unirse!](https://www.coursera.org/learn/natural-language-processing-tensorflow/ungradedLti/jF6xb/important-have-questions-issues-or-ideas-join-our-community-on-discourse)

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