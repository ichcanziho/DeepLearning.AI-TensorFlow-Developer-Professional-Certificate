# Sequences, Time Series and Prediction

[<img src="cover.png" />](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction)

En este curso, aprenderás:

- Resuelva problemas de series temporales y de previsión en TensorFlow

- Prepare los datos para el aprendizaje de series temporales utilizando las mejores prácticas

- Explore cómo pueden utilizarse las RNN y las ConvNets para las predicciones

- Construya un modelo de predicción de manchas solares utilizando datos del mundo real

Si usted es un desarrollador de software que quiere construir algoritmos escalables impulsados por IA, necesita entender 
cómo utilizar las herramientas para construirlos. Esta Specializations le enseñará las mejores prácticas para utilizar 
TensorFlow, un popular marco de trabajo de código abierto para el aprendizaje automático. 

[<img src="certificate.png" />](https://www.coursera.org/account/accomplishments/verify/KNY96VBU28LF)


En este cuarto curso, aprenderá a construir modelos de series temporales en TensorFlow. Primero implementará las mejores 
prácticas para preparar los datos de series temporales. También explorará cómo se pueden utilizar las RNN y las ConvNets 1D 
para la predicción. Por último, ¡aplicará todo lo aprendido a lo largo de la Especialización para construir un modelo de 
predicción de manchas solares utilizando datos del mundo real! 

El curso de Aprendizaje Automático y la Especialización en Aprendizaje Profundo de Andrew Ng enseñan los principios más 
importantes y fundacionales del Aprendizaje Automático y el Aprendizaje Profundo. Esta nueva Especialización en TensorFlow 
de deeplearning.ai le enseña cómo utilizar TensorFlow para implementar esos principios de forma que pueda empezar a construir 
y aplicar modelos escalables a problemas del mundo real. 

Para desarrollar una comprensión más profunda de cómo funcionan las redes neuronales, le recomendamos que realice la 
Especialización en Aprendizaje Profundo.

# INDEX 0

- [Sequences and Prediction](#sequences-and-prediction)
- [Weekly Assignment Create and Predict Synthetic Data](#weekly-assignment-create-and-predict-synthetic-data)
- [Deep Neural Networks for Time Series](#deep-neural-networks-for-time-series)
- [Weekly Assignment Prediction with a DNN](#weekly-assignment-prediction-with-a-dnn)
- [Recurrent Neural Networks for Time Series](#recurrent-neural-networks-for-time-series)
- [Weekly Assignment Using Layers for Sequence Processing](#weekly-assignment-using-layers-for-sequence-processing)
- [Real-world Time Series Data](#real-world-time-series-data)
- [Weekly Assignment Adding Convolutions](#weekly-assignment-adding-convolutions)

# Sequences and Prediction
[<- Return to INDEX 0](#index-0)

## INDEX 1

- [A conversation with Andrew NG W1](#a-conversation-with-andrew-ng-w1)
- [Time Series Examples](#time-series-examples)
- [Machine Learning Applied to Time Series](#machine-learning-applied-to-time-series)
- [Common Patterns in Time Series](#common-patterns-in-time-series)
- [Introduction to Time Series](#introduction-to-time-series)
- [Where to Find the Notebooks for This Course](#where-to-find-the-notebooks-for-this-course)
- [Introduction to Time Series Notebook (Lab 1)](#introduction-to-time-series-notebook-lab-1)
- [Have questions, issues or ideas? Join our Forum!](#have-questions-issues-or-ideas-join-our-forum)
- [Train, Validation and Test Sets](#train-validation-and-test-sets)
- [Metrics for Evaluating Performance](#metrics-for-evaluating-performance)
- [Moving Average and Differencing](#moving-average-and-differencing)
- [Trailing Versus Centered Windows](#trailing-versus-centered-windows)
- [Forecasting](#forecasting)
- [Forecasting Notebooks (Lab 2)](#forecasting-notebooks-lab-2)
- [Week 1 Quiz](#week-1-quiz)
- [Week 1 Wrap Up](#week-1-wrap-up)
- [Lecture Notes Week 1](#lecture-notes-week-1)

La semana 1 de nuestro curso introduce los conceptos fundamentales de las series temporales, cubriendo desde la creación y predicción de datos sintéticos hasta las redes neuronales profundas específicamente diseñadas para el tratamiento de este tipo de datos. A través de asignaciones semanales y explicaciones teóricas, exploraremos cómo las redes neuronales recurrentes pueden ser utilizadas efectivamente para procesar secuencias de tiempo. Además, abordaremos retos del mundo real mediante el uso de datos de series temporales, preparándote para aplicar lo aprendido en escenarios prácticos.

## A conversation with Andrew NG W1
[<- Return to INDEX 1](#index-1)

![img.png](ims%2FW1%2Fimg.png)

En este último curso de la especialización, se profundiza en los modelos de secuencia, centrándose específicamente en series temporales.

- **Concepto de Modelos de Secuencia:**
  - Se exploran modelos de secuencia, que representan datos que cambian con el tiempo, como precios de acciones o condiciones climáticas.
- **Atributos de las Series de Datos:**
  - Se identifican atributos comunes en series de datos, como estacionalidad, tendencias y ruido.
- **Métodos de Predicción:**
  - Se analizan métodos estadísticos y de aprendizaje automático para predecir datos considerando la estacionalidad, tendencias y ruido.
- **Aplicación a la Actividad de Manchas Solares:**
  - Se discute la modelación de la actividad de manchas solares, un fenómeno solar cíclico con implicaciones importantes para la NASA y otras agencias espaciales.
- **Proyección del Curso:**
  - Se inicia con la práctica en datos sintéticos y se concluye aplicando estos conceptos al problema emocionante de la actividad de manchas solares.

¡Comienza esta última fase del curso y adéntrate en la emocionante tarea de modelar las manchas solares!


## Time Series Examples
[<- Return to INDEX 1](#index-1)

![img_1.png](ims%2FW1%2Fimg_1.png)

Bienvenido a este curso de secuencias y predicción, una parte de la especialización de TensorFlow en la práctica. En este curso, nos centraremos en series temporales, donde aprenderás sobre diferentes tipos de series de tiempo antes de profundizar en el uso de datos de series de tiempo. Esta semana, te centrarás en series temporales. Repasaremos algunos ejemplos de diferentes tipos de series temporales, , así como también veremos previsiones básicas a su alrededor. También empezará a preparar datos de series temporales para algoritmos de aprendizaje automático.

Por ejemplo, ¿cómo se dividen los datos de series temporales en conjuntos de entrenamiento, validación y pruebas? Exploraremos algunas mejores prácticas y herramientas alrededor de eso para prepararte para la semana 2, donde empezarás a mirar pronósticos usando un modelo denso, y cómo difiere de predicciones más ingenuas basadas en análisis numérico simple de los datos.

![img_2.png](ims%2FW1%2Fimg_2.png)

En la semana 3, vamos a usar redes neuronales recurrentes para pronosticar series temporales. Veremos los enfoques sin estado y sin estado, capacitación en ventanas de datos, y también tendrás prácticas en la predicción por ti mismo. Finalmente, en la semana 4, agregarás convoluciones a la mezcla y pondrás todo lo que has trabajado juntos para comenzar a pronosticar algunos datos del mundo real, y eso es mediciones de la actividad de manchas solares en los últimos 250 años.

![g1.gif](ims%2FW1%2Fg1.gif)

Así que empecemos con un vistazo a las series temporales, lo que son, y los diferentes tipos de ellas que puedes encontrar. Las series temporales están en todas partes. Puede que los haya visto en los precios de las acciones, pronósticos meteorológicos, tendencias históricas, como la ley de Moore. Aquí, he trazado el número de transistores por milímetro cuadrado, donde agrupé lanzamientos de chips por año, y luego dibujé la línea de tendencia de la ley de Moore desde el primer elemento de datos, donde se puede ver una correlación. 

![img_3.png](ims%2FW1%2Fimg_3.png)

Si quieres algunas correlaciones realmente divertidas, aquí hay una del sitio de Correlaciones Espurias de Tyler Vigen. Esta es una correlación de series temporales de los ingresos totales generados por arcadas de videojuegos versus doctorados en ciencias de la computación otorgados en los Estados Unidos. 

![img_4.png](ims%2FW1%2Fimg_4.png)

Si bien todos estos son bastante familiares, rogaron la pregunta, ¿qué es exactamente una serie de tiempo? Normalmente se define como una secuencia ordenada de valores que suelen estar igualmente espaciados a lo largo del tiempo. Así que, por ejemplo, cada año en mis cartas de leyes de Moore o todos los días en el pronóstico del tiempo. En cada uno de estos ejemplos, hay un valor único en cada paso de tiempo, y como resultado, el término univariado se utiliza para describirlos.

![img_5.png](ims%2FW1%2Fimg_5.png)

También puede encontrar series temporales que tienen varios valores en cada paso de tiempo. Como es de esperar, se llaman Serie Temporal Multivariante. Los gráficos de series temporales multivariadas pueden ser maneras útiles de comprender el impacto de los datos relacionados.

![img_6.png](ims%2FW1%2Fimg_6.png)

Por ejemplo, considere esta tabla de nacimientos frente a muertes en Japón entre 1950 y 2008. Muestra claramente los dos convergentes, pero luego las muertes comienzan a superar nacimientos que conducen a una disminución de la población.
Ahora, aunque podrían tratarse como dos series temporales univariadas separadas, el valor real de los datos se vuelve aparente cuando los mostramos juntos como multivariante.


![img_7.png](ims%2FW1%2Fimg_7.png)

Considere también este gráfico mostrando la temperatura global frente a la concentración de CO2. Como univariados, mostrarían una tendencia, pero cuando se combinan, la correlación es muy fácil de ver agregando más valor a los datos.

![img_8.png](ims%2FW1%2Fimg_8.png)

El movimiento de un cuerpo también se puede trazar como una serie de univariados o como un multivariante combinado. Considere, por ejemplo, el camino de un coche mientras viaja. Un paso de tiempo cero está en una latitud y longitud particulares. Como pasos de tiempo posteriores, estos valores cambiaron en función de la trayectoria del coche. La aceleración del coche, en otras palabras, no se mueve a una velocidad constante, significa que los espacios entre los pasos de tiempo también cambian de tamaño, en este caso cada vez más grande.

![img_9.png](ims%2FW1%2Fimg_9.png)

Pero ¿y si tuviéramos que trazar la dirección del coche como univariado? Basándonos en su rumbo, pudimos ver que la longitud del coche disminuye con el tiempo, pero su latitud aumenta, y como tal obtendremos gráficos como estos.

## Machine Learning Applied to Time Series
[<- Return to INDEX 1](#index-1)

![img_10.png](ims%2FW1%2Fimg_10.png)

Ahora estos son solo algunos ejemplos de los tipos de cosas que se pueden analizar usando series de tiempo. Y casi cualquier cosa que tenga un factor de tiempo en él se puede analizar de esta manera. Entonces, ¿qué tipos de cosas podemos hacer con el aprendizaje automático en series temporales?

![img_11.png](ims%2FW1%2Fimg_11.png)

La primera y más obvia es la predicción de pronósticos basados en los datos. Así que, por ejemplo, con el gráfico de tasas de natalidad y mortalidad para Japón que mostramos antes. Sería muy útil predecir valores futuros para que las agencias gubernamentales puedan planificar la jubilación, la inmigración y otros impactos sociales de estas tendencias.

![img_12.png](ims%2FW1%2Fimg_12.png)

En algunos casos, es posible que también desee proyectar hacia el pasado para ver cómo llegamos a donde estamos ahora. Este proceso se llama imputación. Ahora tal vez quieras tener una idea de cómo habrían sido los datos si hubieras sido capaz de recopilarlos antes de los datos que ya tienes.


![img_13.png](ims%2FW1%2Fimg_13.png)

O es posible que simplemente desee rellenar agujeros en sus datos para qué datos aún no existen. Por ejemplo, en nuestra carta de leyes de Moore desde antes. No hubo datos durante algunos años porque no hubo chips lanzados en esos años, y se pueden ver las brechas aquí.
Pero con la imputación, podemos llenarlos.

![img_14.png](ims%2FW1%2Fimg_14.png)

Además, la predicción de series temporales se puede utilizar para detectar anomalías. Por ejemplo, en los registros del sitio web para que pueda ver la posible denegación de ataques de servicio apareciendo como un aumento en las series temporales como esta.

![img_15.png](ims%2FW1%2Fimg_15.png)

La otra opción es analizar las series temporales para detectar patrones en ellas que determinan lo que generó la serie misma. Un ejemplo clásico de esto es analizar ondas sonoras para detectar palabras en ellas que pueden ser utilizadas como una red neuronal para el reconocimiento de voz. Aquí, por ejemplo, puede ver cómo una onda de sonido se divide en palabras. Usando el aprendizaje automático, se hace posible entrenar una red neuronal basada en la serie temporal para reconocer palabras o subpalabras.

## Common Patterns in Time Series
[<- Return to INDEX 1](#index-1)

![img_16.png](ims%2FW1%2Fimg_16.png)

Las series temporales vienen en todas las formas y tamaños, pero hay una serie de patrones muy comunes. Así que es útil reconocerlos cuando los ves. Durante los próximos minutos vamos a echar un vistazo a algunos ejemplos. 

![img_17.png](ims%2FW1%2Fimg_17.png)

La primera es tendencia, donde las series temporales tienen una dirección específica en la que se están moviendo. Como pueden ver en el ejemplo de la Ley de Moore que mostramos anteriormente, esta es una tendencia hacia arriba.

![img_18.png](ims%2FW1%2Fimg_18.png)

Otro concepto es la estacionalidad, que se ve cuando los patrones se repiten a intervalos predecibles. Por ejemplo, eche un vistazo a este gráfico que muestra usuarios activos en un sitio web para desarrolladores de software. Sigue un patrón muy distinto de caídas regulares. ¿Puedes adivinar lo que son? Bueno, ¿y si te dijera si estaba arriba para cinco unidades y luego abajo para dos? Entonces se podría decir que muy claramente cae en los fines de semana cuando menos gente está trabajando y por lo tanto muestra estacionalidad.

Otras series de temporada podrían ser sitios de compras que pico en fines de semana o sitios de deportes que pico en varios momentos a lo largo del año, como el draft o el día de apertura, el All-Star día playoffs y tal vez el juego del campeonato. 

![img_19.png](ims%2FW1%2Fimg_19.png)

Por supuesto, algunas series temporales pueden tener una combinación de tendencia y estacionalidad como muestra este gráfico. Hay una tendencia alcista global pero hay picos y valles locales.

![img_20.png](ims%2FW1%2Fimg_20.png)

Pero, por supuesto, también hay algunos que son probablemente no predecibles en absoluto y solo un conjunto completo de valores aleatorios produciendo lo que típicamente se llama ruido blanco. No hay mucho que puedas hacer con este tipo de datos. 

![img_21.png](ims%2FW1%2Fimg_21.png)

Pero entonces considere esta serie temporal. No hay tendencia y no hay estacionalidad. Los picos aparecen en marcas de tiempo aleatorias. No se puede predecir cuándo va a pasar a continuación o cuán fuertes serán. Pero claramente, toda la serie no es aleatoria. Entre los picos hay un tipo muy determinista de descomposición. 

![img_22.png](ims%2FW1%2Fimg_22.png)

Podemos ver aquí que el valor de cada paso de tiempo es 99 por ciento del valor de el paso de tiempo anterior más un pico ocasional. Esta es una serie cronológica automática correlacionada. Es decir, se correlaciona con una copia retrasada de sí misma a menudo llamada un retraso.
Este ejemplo se puede ver en el lag uno hay una fuerte autocorrelación. 

![img_23.png](ims%2FW1%2Fimg_23.png)

A menudo, una serie de tiempo como esta se describe como tener memoria ya que los pasos dependen de los anteriores. Los picos que son impredecibles son a menudo llamados Innovaciones. En otras palabras, no se pueden predecir basándose en valores pasados. 

![img_24.png](ims%2FW1%2Fimg_24.png)

Otro ejemplo es aquí donde hay múltiples autocorrelaciones, en este caso, en los pasos uno y 50. El retardo uno autocorrelación da estos retrasos exponenciales muy rápidos a corto plazo, y el 50 da el pequeño equilibrio después de cada pico. 

![img_25.png](ims%2FW1%2Fimg_25.png)

Las series temporales que encontrarás en la vida real probablemente tienen un poco de cada una de estas características: tendencia, estacionalidad, autocorrelación y ruido. Como hemos aprendido, un modelo de aprendizaje automático está diseñado para detectar patrones, y cuando detectamos patrones podemos hacer predicciones. En su mayor parte, esto también puede funcionar con series temporales excepto por el ruido que es impredecible.

![img_26.png](ims%2FW1%2Fimg_26.png)

Pero debemos reconocer que este asume que los patrones que existieron en el pasado continuarán por supuesto en el futuro. Por supuesto, las series de tiempo de la vida real no siempre son tan simples. Su comportamiento puede cambiar drásticamente con el tiempo.

![img_27.png](ims%2FW1%2Fimg_27.png)

Por ejemplo, esta serie temporal tuvo una tendencia positiva y una estacionalidad clara hasta el paso de tiempo 200. Pero entonces algo sucedió para cambiar su comportamiento por completo. Si esto fuera acciones, precio entonces tal vez fue una gran crisis financiera o un gran escándalo o tal vez un avance tecnológico disruptivo causando un cambio masivo.

![img_28.png](ims%2FW1%2Fimg_28.png)

Después de eso, la serie de tiempo comenzó a tendencia a la baja sin ninguna estacionalidad clara. Normalmente llamaremos a esto una serie temporal no estacionaria. Para predecir sobre esto podríamos entrenar por un período limitado de tiempo. Por ejemplo, aquí donde doy sólo los últimos 100 pasos. Probablemente obtendrá un mejor rendimiento que si se hubiera entrenado en toda la serie temporal. Pero eso es romper el molde para la máquina típica, aprendiendo donde siempre asumimos que más datos es mejor.

![img_29.png](ims%2FW1%2Fimg_29.png)

Pero para la predicción de series de tiempo realmente depende de la serie de tiempo. Si es estacionario, lo que significa que su comportamiento no cambia con el tiempo, entonces genial. Cuantos más datos tengas, mejor. Pero si no es estacionario entonces la ventana de tiempo óptima que debe usar para el entrenamiento variará. Idealmente, nos gustaría que pudiera tener en cuenta toda la serie y generar una predicción de lo que podría suceder a continuación. 

Como puedes ver, esto no siempre es tan simple como podrías pensar dado un cambio drástico como el que vemos aquí. Así que eso es algo de lo que vas a estar viendo en este curso. Pero vamos a empezar por revisar un libro de trabajo que genera secuencias como las que vio en este video. Después de eso intentaremos predecir algunas de estas secuencias sintetizadas como una práctica antes de pasar a los datos del mundo real.

## Introduction to Time Series
[<- Return to INDEX 1](#index-1)

Primero, vamos a importar las librerías necesarias para trabajar con nuestros datos y visualizaciones:

```python
import matplotlib.pyplot as plt
import numpy as np
```

Estas son las bibliotecas básicas que vamos a utilizar. `matplotlib.pyplot` nos permitirá crear gráficas para visualizar nuestras series de tiempo, mientras que `numpy` se utilizará para realizar operaciones matemáticas y manejar nuestros datos.

---

### Utilidades para Graficar

Aquí definimos una función útil para graficar nuestras series de tiempo:

```python
def plot_series(time, series, format="-", start=0, end=None, label=None):
    """
    Visualiza datos de series de tiempo.

    Parámetros:
      time (array de int) - contiene los pasos de tiempo
      series (array de int) - contiene las mediciones para cada paso de tiempo
      format (string) - estilo de línea al graficar
      start (int) - primer paso de tiempo a graficar
      end (int) - último paso de tiempo a graficar
      label (lista de strings)- etiqueta para la línea
    """

    plt.figure(figsize=(10, 6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Tiempo")
    plt.ylabel("Valor")
    if label:
      plt.legend(fontsize=14, labels=label)
    plt.grid(True)
    plt.show()
```

Esta función es crucial para visualizar las series de tiempo, permitiéndonos comprender mejor la estructura y comportamiento de nuestros datos.

---

### Tendencia

La tendencia describe la inclinación general de los valores a aumentar o disminuir a medida que avanza el tiempo. Esto se puede visualizar fácilmente generando datos que siguen una línea recta.

```python
def trend(time, slope=0):
    """
    Genera datos sintéticos que siguen una línea recta dada un valor de pendiente.

    Parámetros:
      time (array de int) - contiene los pasos de tiempo
      slope (float) - determina la dirección y la inclinación de la línea

    Devuelve:
      series (array de float) - mediciones que siguen una línea recta
    """

    series = slope * time
    return series
```

Generemos una serie de tiempo que tenga una tendencia ascendente:

```python
time = np.arange(365)
slope = 0.1
series = trend(time, slope)
plot_series(time, series, label=[f'pendiente={slope}'])
```
![img_31.png](ims%2FW1%2Fimg_31.png)
Este concepto es esencial para entender cómo se comportan los datos a lo largo del tiempo y para hacer predicciones.

---

### Estacionalidad

La estacionalidad se refiere a patrones recurrentes en intervalos de tiempo regulares.

```python
def seasonal_pattern(season_time):
    """
    Solo un patrón arbitrario, puedes cambiarlo si deseas.

    ...

    Devuelve:
      data_pattern (array de float) - contiene valores de medición revisados
                                      según el patrón definido
    """
    # Generate the values using an arbitrary pattern
    data_pattern = np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

    return data_pattern
```

Esta función nos ayudará a crear patrones estacionales que repiten ciclos a lo largo del tiempo:

```python
def seasonality(time, period, amplitude=1, phase=0):
    """
    Repite el mismo patrón en cada periodo.
    ...
    """

    season_time = ((time + phase) % period) / period
    data_pattern = amplitude * seasonal_pattern(season_time)
    return data_pattern
```

Podemos usar estas funciones para generar series de tiempo con patrones estacionales claros.

```python
# Generate time steps
time = np.arange(4 * 365 + 1)

# Define the parameters of the seasonal data
period = 365
amplitude = 40

# Generate the seasonal data
series = seasonality(time, period=period, amplitude=amplitude)

# Plot the results
plot_series(time, series)
```

![img_32.png](ims%2FW1%2Fimg_32.png)


```python
# Define seasonal parameters
slope = 0.05
period = 365
amplitude = 40

# Generate the data
series = trend(time, slope) + seasonality(time, period=period, amplitude=amplitude)

# Plot the results
plot_series(time, series)
```

![img_33.png](ims%2FW1%2Fimg_33.png)

---

### Ruido

En la práctica, pocas series de tiempo en la vida real son completamente suaves. Por lo general, tienen ruido superpuesto a la señal.

```python
def noise(time, noise_level=1, seed=None):
    """
    Genera una señal ruidosa distribuida de manera normal.

    ...
    """

    rnd = np.random.RandomState(seed)
    noise = rnd.randn(len(time)) * noise_level
    return noise
```

Agregar ruido a nuestras series de tiempo las hace más realistas y nos permite practicar cómo manejar datos imperfectos.

```python
# Define noise level
noise_level = 5

# Generate noisy signal
noise_signal = noise(time, noise_level=noise_level, seed=42)

# Plot the results
plot_series(time, noise_signal)
```

![img_34.png](ims%2FW1%2Fimg_34.png)

```python
# Add the noise to the time series
series += noise_signal

# Plot the results
plot_series(time, series)
```

![img_35.png](ims%2FW1%2Fimg_35.png)
---

### Autocorrelación

Las series de tiempo también pueden ser autocorrelacionadas, lo que significa que las mediciones en un momento dado son una función de los pasos de tiempo anteriores.

```python
def autocorrelation(time, amplitude, seed=None):
    """
    Genera datos autocorrelacionados.

    ...

    Devuelve:
      ar (array de float) - datos autocorrelacionados
    """
    # Initialize random number generator
    rnd = np.random.RandomState(seed)

    # Initialize array of random numbers equal to the length
    # of the given time steps plus 50
    ar = rnd.randn(len(time) + 50)

    # Set first 50 elements to a constant
    ar[:50] = 100

    # Define scaling factors
    phi1 = 0.5
    phi2 = -0.1

    # Autocorrelate element 51 onwards with the measurement at
    # (t-50) and (t-30), where t is the current time step
    for step in range(50, len(time) + 50):
        ar[step] += phi1 * ar[step - 50]
        ar[step] += phi2 * ar[step - 33]

    # Get the autocorrelated data and scale with the given amplitude.
    # The first 50 elements of the original array is truncated because
    # those are just constant and not autocorrelated.
    ar = ar[50:] * amplitude

    return ar
```

Este concepto es fundamental en el análisis de series de tiempo, ya que muchos métodos de predicción dependen de la autocorrelación para hacer pronósticos precisos.

```python
# Use time steps from previous section and generate autocorrelated data
series = autocorrelation(time, amplitude=10, seed=42)

# Plot the first 200 elements to see the pattern more clearly
plot_series(time[:200], series[:200])
```

![img_36.png](ims%2FW1%2Fimg_36.png)

---

### Series de Tiempo No Estacionarias

Es posible que las series de tiempo rompan un patrón esperado debido a eventos significativos, alterando la tendencia o comportamiento estacional de los datos.

```python
# Generate data with positive trend
series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)

# Generate data with negative trend
series2 = autocorrelation(time, 5, seed=42) + seasonality(time, period=50, amplitude=2) + trend(time, -1) + 550

# Splice the downward trending data into the first one at time step = 200
series[200:] = series2[200:]

# Plot the result
plot_series(time[:300], series[:300])
```

![img_37.png](ims%2FW1%2Fimg_37.png)

En situaciones como esta, puede ser beneficioso entrenar el modelo en los datos más recientes, ya que estos ofrecen una señal predictiva más fuerte para los pasos de tiempo futuros.

## Where to Find the Notebooks for This Course
[<- Return to INDEX 1](#index-1)

Todos los cuadernos de este curso pueden ejecutarse en Google Colab o en Coursera Labs. **No necesita tener configurado un entorno local para seguir los ejercicios de codificación.** Se le llevará a Google Colab para los laboratorios no calificados, mientras que para las tareas, se le llevará automáticamente a Coursera Labs. 

Sin embargo, si desea ejecutarlos en su máquina local, los laboratorios no calificados y las asignaciones para cada semana se pueden encontrar en este 
[repositorio de Github](https://github.com/https-deeplearning-ai/tensorflow-1-public)
 bajo la carpeta **C4**. Si ya tiene git instalado en su ordenador, puede clonarlo con este comando:

> git clone https://github.com/https-deeplearning-ai/tensorflow-1-public

Si no, por favor siga las guías 
[aquí](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
 para instalar git en su sistema operativo. Una vez que haya clonado el repositorio, puede hacer un `git pull` de vez en cuando para asegurarse de que recibe las últimas actualizaciones de los cuadernos.

Necesitará estos paquetes si va a ejecutar los cuadernos localmente:

````requirements
tensorflow==2.7.0
numpy==1.20.1
matplotlib==3.2.2
````

## Introduction to Time Series Notebook (Lab 1)
[<- Return to INDEX 1](#index-1)


Aquí [C4_W1_Lab_1_time_series.ipynb](notebooks%2FW1%2FC4_W1_Lab_1_time_series.ipynb)
 está el enlace al cuaderno que Laurence acaba de mostrar en el screencast anterior.

![img_30.png](ims%2FW1%2Fimg_30.png)

## Have questions, issues or ideas? Join our Forum!
[<- Return to INDEX 1](#index-1)

¡Hola alumno!

¿Emocionado por este curso? Únase a nuestro Foro para:

- Chatear con otros: Compartir ideas, hacer preguntas y discutir sobre IA.

- Trabajar juntos: Colabore en proyectos de IA y construya algo impresionante.

- Manténgase al día: Reciba actualizaciones sobre cursos, eventos y noticias de IA.

Haga clic en 
[este enlace](https://community.deeplearning.ai/invites/ejnUJ7tNu2)
 para crear su cuenta gratuita y conectar con la comunidad global de IA

- El equipo de DeepLearning.IA


## Train, Validation and Test Sets
[<- Return to INDEX 1](#index-1)

![img_38.png](ims%2FW1%2Fimg_38.png)

En los videos anteriores de esta semana, viste todos los diferentes factores que conforman el comportamiento de una serie de tiempo. Ahora vamos a empezar a buscar técnicas, dado lo que sabemos, que pueden ser usadas para luego pronosticar esa serie temporal.

![img_39.png](ims%2FW1%2Fimg_39.png)

Comencemos con esta serie de tiempo que contiene, tendencia estacionalidad, y ruido y eso es lo suficientemente realista por ahora.

![img_40.png](ims%2FW1%2Fimg_40.png)

Podríamos, por ejemplo, tomar el último valor y asumir que el siguiente valor será el mismo, y esto se llama predicción ingenua. He ampliado una parte del conjunto de datos aquí para mostrar eso en acción. Podemos hacer eso para obtener una línea de base por lo menos, y créanlo o no, esa línea de base puede ser bastante buena. 


![img_41.png](ims%2FW1%2Fimg_41.png)

Pero, ¿cómo se mide el rendimiento? Para medir el rendimiento de nuestro modelo de previsión,. Normalmente queremos dividir las series temporales en un período de entrenamiento, un período de validación y un período de prueba. Esto se llama partición fija. 

Si la serie temporal tiene alguna estacionalidad, generalmente desea asegurarse de que cada período contenga un número entero de estaciones. Por ejemplo, un año, dos años, o tres años, si la serie temporal tiene una estacionalidad anual. Usted generalmente no quiere un año y medio, o de lo contrario algunos meses estarán representados más que otros. 

Aunque esto puede parecer un poco diferente de la prueba de validación de entrenamiento, que puede estar familiarizado con los conjuntos de datos de series no temporales. Donde acabas de elegir valores aleatorios del corpus para hacer los tres, deberías ver que el impacto es efectivamente el mismo. 

A continuación entrenarás a tu modelo en el periodo de entrenamiento, y lo evaluarás en el periodo de validación. Aquí es donde puede experimentar para encontrar la arquitectura adecuada para la formación. Y trabaje en él y en sus hiperparámetros, hasta que obtenga el rendimiento deseado, medido usando el conjunto de validación. 

A menudo, una vez que hayas hecho eso, puedes volver a entrenar usando los datos de validación de la capacitación y Y luego prueba en el período de prueba para ver si tu modelo funcionará igual de bien. Y si lo hace, entonces usted podría dar el paso inusual de reciclaje de nuevo, usando también los datos de prueba. Pero, ¿por qué harías eso? Bueno, es porque los datos de prueba son los datos más cercanos que tiene al punto actual en el tiempo. Y como tal, a menudo es la señal más fuerte para determinar valores futuros. 


![img_42.png](ims%2FW1%2Fimg_42.png)

Si su modelo no está entrenado usando esos datos, también, entonces puede que no sea óptimo. Debido a esto, en realidad es bastante común renunciar a un conjunto de pruebas todos juntos. Y sólo entrenar, usando un período de entrenamiento y un período de validación, y el conjunto de pruebas es en el futuro. Vamos a seguir parte de esa metodología en este curso. 

![img_43.png](ims%2FW1%2Fimg_43.png)

La partición fija como esta es muy simple y muy intuitiva, pero también hay otra manera. Comenzamos con un corto período de entrenamiento, y poco a poco lo aumentamos, dice por un día a la vez, o por una semana a la vez. En cada iteración, entrenamos al modelo en un período de entrenamiento. 

 Y lo usamos para pronosticar el día siguiente, o la semana siguiente, en el período de validación. Y esto se llama partición roll-forward. Podría verlo como haciendo particiones fijas varias veces, y luego refinando continuamente el modelo como tal. 

Con el propósito de aprender la predicción de series de tiempo en este curso, aprenderá el código general para hacer la predicción de series. Que entonces podrías aplicarte a un escenario roll-forward, pero nuestro enfoque va a estar en la partición fija.

## Metrics for Evaluating Performance
[<- Return to INDEX 1](#index-1)

![img_44.png](ims%2FW1%2Fimg_44.png)

Una vez que tengamos un modelo y un período, entonces podemos evaluar el modelo en él, y necesitaremos una métrica para calcular su rendimiento.
Así que comencemos simplemente calculando los errores, que es la diferencia entre los valores previstos de nuestro modelo y los valores reales durante el período de evaluación.

La métrica más común para evaluar el rendimiento de predicción de un modelo es el error cuadrado medio o `mse` donde cuadramos los 
errores y luego calculamos su media. Bueno, ¿por qué lo cuadraríamos? Bueno, la razón de esto es deshacerse de los valores 
negativos. 

Entonces, por ejemplo, si nuestro error fue dos por encima del valor, entonces serán dos, pero si fueron dos por debajo del valor, entonces será menos dos. Estos errores podrían entonces efectivamente cancelarse unos a otros, lo cual será incorrecto porque tenemos dos errores y no ninguno. Pero si cuadramos el error de valor antes de analizar, entonces ambos errores serían cuadrados a cuatro, no cancelándose unos a otros y siendo efectivamente iguales.


Y si queremos que la media del cálculo de nuestros errores sea de la misma escala que los errores originales, entonces solo obtenemos su raíz cuadrada, dándonos una raíz significa error cuadrado o rmse. 

![img_45.png](ims%2FW1%2Fimg_45.png)

Otra métrica común y uno de mis favoritos es el error absoluto medio o mae, y también se llama la principal desviación absoluta o loco. Y en este caso, en lugar de cuadrar para deshacerse de los negativos, solo usa su valor absoluto. Esto no penaliza los errores grandes tanto como lo hace el mse. Dependiendo de su tarea, puede que prefiera la mae o la mse.

Por ejemplo, si los errores grandes son potencialmente peligrosos y le cuestan mucho más que los errores más pequeños, entonces usted puede preferir el mse. Pero si tu ganancia o tu pérdida es proporcional al tamaño del error, entonces el mae puede ser mejor.

Además, puede medir el porcentaje medio de error absoluto o mape, esta es la relación media entre el error absoluto y el valor absoluto, esto da una idea del tamaño de los errores en comparación con los valores.

![img_46.png](ims%2FW1%2Fimg_46.png)

Si miramos nuestros datos, podemos medir el MAE usando un código como este. 
Las bibliotecas de métricas de keras incluyen un MAE que se puede llamar así. Con los datos sintéticos que mostramos anteriormente, estamos obteniendo alrededor de 5.93, consideremos que nuestra línea de base.



## Moving Average and Differencing
[<- Return to INDEX 1](#index-1)

![img_47.png](ims%2FW1%2Fimg_47.png)

Un método común y muy simple de predicción es calcular una media móvil. La idea aquí es que la línea amarilla es una trama de la media de los valores azules sobre un período fijo llamado ventana de promediación, por ejemplo, 30 días. Ahora esto elimina bien mucho ruido y nos da una curva que emula más o menos la serie original, pero no anticipa tendencia o estacionalidad.

Dependiendo de la hora actual, es decir, el período después del cual desea para pronosticar para el futuro, en realidad puede terminar siendo peor que un pronóstico ingenuo. En este caso, por ejemplo, obtuve un error absoluto medio de aproximadamente 7.14. 

![img_48.png](ims%2FW1%2Fimg_48.png)

Un método para evitar esto es eliminar la tendencia y la estacionalidad de la serie temporal con una técnica llamada diferenciación. Así que en lugar de estudiar la serie temporal en sí, estudiamos la diferencia entre el valor en el tiempo T y el valor en un período anterior. Dependiendo de la hora de sus datos, ese período podría ser un año, un día, un mes o lo que sea.


![img_49.png](ims%2FW1%2Fimg_49.png)

Veamos un año antes. Así que para estos datos, en el tiempo T menos 365, obtendremos esta diferencia de series temporales que no tiene tendencia ni estacionalidad. Entonces podemos utilizar una media móvil para pronosticar esta serie temporal que nos da estas previsiones. 

![img_50.png](ims%2FW1%2Fimg_50.png)

Pero estos son sólo pronósticos para la serie de tiempo de diferencia, no la serie de tiempo original. Para obtener las previsiones finales para la serie temporal original, sólo tenemos que añadir el valor en el tiempo T menos 365, y obtendremos estas previsiones. Se ven mucho mejor, ¿no? Si medimos el error absoluto medio en el período de validación, obtenemos aproximadamente 5.8. Así que es un poco mejor que el pronóstico ingenuo pero no tremendamente mejor. 

![img_51.png](ims%2FW1%2Fimg_51.png)

Usted puede haber notado que nuestra media móvil eliminó una gran cantidad de ruido , pero nuestras previsiones finales siguen siendo bastante ruidosas. ¿De dónde viene ese ruido? Bueno, eso viene de los valores pasados que agregamos de nuevo en nuestras previsiones. Así que podemos mejorar estos pronósticos eliminando también el ruido pasado usando una media móvil sobre eso. 

Si hacemos eso, obtenemos pronósticos mucho más suaves. De hecho, esto nos da un error cuadrado medio sobre el período de validación de solo 4,5. Ahora eso es mucho mejor que todos los métodos anteriores. De hecho, ya que se genera la serie, podemos calcular que un modelo perfecto dará un error absoluto medio de aproximadamente cuatro debido al ruido. Aparentemente, con este enfoque, no estamos demasiado lejos de lo óptimo. Ten esto en cuenta antes de que te apresures al aprendizaje profundo. Los enfoques simples a veces pueden funcionar bien.

## Trailing Versus Centered Windows
[<- Return to INDEX 1](#index-1)

![img_53.png](ims%2FW1%2Fimg_53.png)

Tenga en cuenta que cuando usamos la ventana final al calcular la media móvil de los valores actuales de t menos 32, t menos uno. Pero cuando usamos una ventana centrada para calcular la media móvil de valores pasados de hace un año, eso es t menos un año menos cinco días, a t menos un año más cinco días.

Entonces los promedios móviles que utilizan ventanas centradas pueden ser más precisos que usar ventanas finales. Pero no podemos usar ventanas centradas para suavizar los valores actuales ya que no conocemos valores futuros. Sin embargo, para suavizar los valores pasados podemos permitirnos usar ventanas centradas.

![img_52.png](ims%2FW1%2Fimg_52.png)

ale, así que ahora hemos mirado a algunos métodos estadísticos para predecir los siguientes valores en una serie temporal. En el siguiente video, echarás un vistazo a un screencast de esta predicción en acción. Una vez que hayas hecho el pronóstico estadístico, el siguiente paso, por supuesto, será aplicar las técnicas de aprendizaje automático has estado aprendiendo todo el tiempo en TensorFlow y lo harás la próxima semana.

## Forecasting
[<- Return to INDEX 1](#index-1)

> Nota: se reutilizarán las funciones del notebook pasado pltear las gráficas, generar ruido, temporalidad etc

#### Generando datos sintéticos:

```python
# Parameters
time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

# Update with noise
series += noise(time, noise_level, seed=42)

# Plot the results
plot_series(time, series)
```

![img_54.png](ims%2FW1%2Fimg_54.png)

#### Cortando el dataset:

Generar train y test es sumamente simple en este contexto:

```python
# Define the split time
split_time = 1000

# Get the train set
time_train = time[:split_time]
x_train = series[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]
```

#### Naive Forecast

vamos a predecir el valor actual como una copia del valor anterior:

```python
# Generate the naive forecast
naive_forecast = series[split_time - 1:-1]

# Define time step
time_step = 100

# Print values
print(f'ground truth at time step {time_step}: {x_valid[time_step]}')
print(f'prediction at time step {time_step + 1}: {naive_forecast[time_step + 1]}')
# Plot the results
plot_series(time_valid, (x_valid, naive_forecast))
```

```commandline
ground truth at time step 100: 109.84197998046875
prediction at time step 101: 109.84197998046875
```

![img_55.png](ims%2FW1%2Fimg_55.png)

Resultados:

```python
print(tf.keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

61.827534
5.937908
```

#### Moving Average

```python
def moving_average_forecast(series, window_size):
    """Generates a moving average forecast

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to compute the average for

    Returns:
      forecast (array of float) - the moving average forecast
    """

    # Initialize a list
    forecast = []

    # Compute the moving average based on the window size
    for time in range(len(series) - window_size):
      forecast.append(series[time:time + window_size].mean())

    # Convert to a numpy array
    forecast = np.array(forecast)

    return forecast

# Generate the moving average forecast
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

# Plot the results
plot_series(time_valid, (x_valid, moving_avg))
```

![img_56.png](ims%2FW1%2Fimg_56.png)

```python
print(tf.keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
106.674576
7.142419
```

#### Differencing

```python
# Subtract the values at t-365 from original series
diff_series = (series[365:] - series[:-365])

# Truncate the first 365 time steps
diff_time = time[365:]

# Plot the results
plot_series(diff_time, diff_series)
```

![img_57.png](ims%2FW1%2Fimg_57.png)

```python
# Generate moving average from the time differenced dataset
diff_moving_avg = moving_average_forecast(diff_series, 30)

# Slice the prediction points that corresponds to the validation set time steps
diff_moving_avg = diff_moving_avg[split_time - 365 - 30:]

# Slice the ground truth points that corresponds to the validation set time steps
diff_series = diff_series[split_time - 365:]

# Plot the results
plot_series(time_valid, (diff_series, diff_moving_avg))
```

![img_58.png](ims%2FW1%2Fimg_58.png)

```python
# Add the trend and seasonality from the original series
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

# Plot the results
plot_series(time_valid, (x_valid, diff_moving_avg_plus_past))
```

![img_59.png](ims%2FW1%2Fimg_59.png)

```python
print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
53.764587
5.9032416
```

#### Smoothing

```python
# Smooth the original series before adding the time differenced moving average
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-359], 11) + diff_moving_avg

# Plot the results
plot_series(time_valid, (x_valid, diff_moving_avg_plus_smooth_past))
```

![img_60.png](ims%2FW1%2Fimg_60.png)

```python
 # Compute the metrics
print(tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
34.315723
4.6053295
```

## Forecasting Notebooks (Lab 2)
[<- Return to INDEX 1](#index-1)

Aquí
 tiene el enlace al cuaderno que muestra los métodos estadísticos en la previsión.

## Week 1 Quiz
[<- Return to INDEX 1](#index-1)

1. **Question 1**
What is an example of a Univariate time series?

   - [ ] Hour by hour weather
   - [ ] Baseball scores
   - [x] Hour by hour temperature
   - [ ] Fashion items

   > **Correct!** A univariate time series focuses on a single variable over time, like the temperature measured across different hours.

2. **Question 2**
What is an example of a Multivariate time series?

   - [x] Hour by hour weather
   - [ ] Baseball scores
   - [ ] Hour by hour temperature
   - [ ] Fashion items

   > **Correct!** Multivariate time series involve multiple variables that are observed over time. Hour-by-hour weather data can include temperature, humidity, wind speed, etc., making it a perfect example.

3. **Question 3**
What is imputed data?

   - [x] A projection of unknown (usually past or missing) data
   - [ ] Data that has been withheld for various reasons
   - [ ] A bad prediction of future data
   - [ ] A good prediction of future data

   > **Correct!** Imputed data refers to filling in missing or incomplete data with substituted values, often projected from known data.

4. **Question 4**
A sound wave is a good example of time series data.

   - [ ] False
   - [x] True

   > **Correct!** Sound wave data is indeed a prime example of time series data as it records the amplitude variations of sound over time.

5. **Question 5**
What is Seasonality?

   - [x] A regular change in shape of the data
   - [ ] Weather data
   - [ ] Data that is only available at certain times of the year
   - [ ] Data aligning to the 4 seasons of the calendar

   > **Correct!** Seasonality refers to predictable and regular patterns or fluctuations that are recurrent over specific time periods within data.

6. **Question 6**
What is a trend?

   - [ ] An overall consistent upward direction for data
   - [ ] An overall consistent flat direction for data
   - [ ] An overall consistent downward direction for data
   - [x] An overall direction for data regardless of direction

   > **Correct!** A trend in time series data shows the general direction in which the data moves over a long period, irrespective of the direction it takes.

7. **Question 7**
In the context of time series, what is noise?

   - [x] Unpredictable changes in time series data
   - [ ] Sound waves forming a time series
   - [ ] Data that doesn’t have seasonality
   - [ ] Data that doesn’t have a trend

   > **Correct!** Noise refers to the random fluctuation or variability in a time series that cannot be attributed to trend or seasonal effects.

8. **Question 8**
What is autocorrelation?

   - [ ] Data that doesn’t have noise
   - [ ] Data that automatically lines up seasonally
   - [ ] Data that automatically lines up in trends
   - [x] Data that follows a predictable shape, even if the scale is different

   > **Correct!** Autocorrelation occurs when a time series is correlated with a lagged version of itself, indicating that the data follows a predictable pattern over time.

9. **Question 9**
What is a non-stationary time series?

   - [x] One that has a disruptive event breaking trend and seasonality.
   - [ ] One that is consistent across all seasons.
   - [ ] One that has a constructive event forming trend and seasonality.
   - [ ] One that moves seasonally.

   > **Correct!** A non-stationary time series is characterized by having statistical properties such as mean, variance, and autocorrelation that change over time, often due to disruptions breaking existing trends and seasonality.

## Week 1 Wrap Up
[<- Return to INDEX 1](#index-1)

Esta semana ha explorado la naturaleza de los datos de series temporales y ha visto algunos de sus atributos más comunes, como la estacionalidad y la tendencia. También ha visto algunos métodos estadísticos para predecir datos de series temporales. 

La semana que viene, empezará a estudiar el uso de las DNN para la clasificación de series temporales, incluida la importante tarea de comprender cómo dividir una serie temporal en datos de entrenamiento y de validación.

## Lecture Notes Week 1
[<- Return to INDEX 1](#index-1)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C4_W1.pdf](notes%2FC4_W1.pdf)
.

# Weekly Assignment Create and Predict Synthetic Data
[<- Return to INDEX 0](#index-0)

## Assignment Troubleshooting Tips
[<- Return to INDEX 1](#index-1)

He aquí algunas directrices generales antes de entregar sus tareas en este curso. Téngalas en cuenta no sólo para la tarea de esta semana, sino también para las siguientes:

1. Asegúrese de guardar su trabajo antes de hacer clic en el botón `Submit`. De lo contrario, es posible que aparezca un 
mensaje de error como el de la celda de código siguiente. Recuerde que todo lo que tiene que rellenar dentro de las 
funciones calificadas se inicializa en `None`.

   ```commandline
   Failed test case: x has incorrect type.
   Expected:
   some.Type,
   but got:
   <class 'NoneType'>.
   ```

2. Por favor, no cambie el nombre del cuaderno. El calificador buscará el nombre de archivo original y sus metadatos asociados, por lo que debe trabajar en el archivo que se abre automáticamente al hacer clic en el botón `Launch Notebook`. Si intenta enviar un cuaderno renombrado, es posible que también aparezca un error como el que se muestra arriba.

3. Por favor, no modifique ningún código fuera de las etiquetas `START CODE HERE`  y `END CODE HERE`. Su solución sólo debe colocarse entre estos marcadores para garantizar una calificación correcta. Modificar los parámetros de las funciones y otras celdas de prueba probablemente romperá el calificador. Si desea experimentar con ellas, puede hacerlo después de haber superado con éxito la tarea.

4. Después de seguir los consejos anteriores y el calificador le sigue dando 0/100, es posible que los metadatos necesarios para la calificación estén dañados. Por favor, obtenga un nuevo cuaderno de laboratorio refrescando su espacio de trabajo
([instrucciones aquí](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/F77WP/optional-downloading-your-notebook-and-refreshing-your-workspace)
). A continuación, copie sus soluciones en el nuevo cuaderno. Asegúrese de que todas las celdas siguen funcionando como se espera y, a continuación, vuelva a enviarlas.

5. Si tiene más preguntas, por favor cree un tema en la comunidad Discourse en lugar de los foros de discusión de Coursera. Puede unirse 
[siguiendo las instrucciones aquí](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/ungradedLti/smWBG/important-have-questions-issues-or-ideas-join-our-community-on-discourse)
. Obtendrá ayuda allí más rápidamente porque varios mentores y sus compañeros de aprendizaje están supervisando los mensajes. Sólo asegúrese de crear el tema en la categoría correcta del curso.



## Downloading your Notebooks and Refreshing your Workspace
[<- Return to INDEX 1](#index-1)

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


## Working with Generated Time Series
[<- Return to INDEX 1](#index-1)

En clase, usted creó una serie de datos sintéticos e hizo predicciones utilizando métodos estadísticos. Este cuaderno contiene un patrón diferente para la serie de datos. Vea si puede hacer predicciones para ella similares al proceso mostrado en las clases

> ## Respuesta: [C4W1_Assignment.ipynb](notebooks%2FW1%2Fassigment%201%2FC4W1_Assignment.ipynb)

# Deep Neural Networks for Time Series
[<- Return to INDEX 0](#index-0)

## INDEX 2

- [A Conversation with Andrew Ng W2](#a-conversation-with-andrew-ng-w2)
- [Preparing Features and Labels Notebook (Lab 1)](#preparing-features-and-labels-notebook-lab-1)
- [Preparing Features and Labels](#preparing-features-and-labels)
- [Preparing Features and Labels (screencast)](#preparing-features-and-labels-screencast)
- [Single Layer Network Notebook (Lab 2)](#single-layer-network-notebook-lab-2)
- [Feeding Windowed Dataset into Neural Network](#feeding-windowed-dataset-into-neural-network)
- [Single Layer Neural Network](#single-layer-neural-network)
- [Machine Learning on Time Windows](#machine-learning-on-time-windows)
- [Prediction](#prediction)
- [More on Single Layer Neural Network](#more-on-single-layer-neural-network)
- [Deep Neural Network Notebook (Lab 3)](#deep-neural-network-notebook-lab-3)
- [Deep Neural Network Training, Tuning and Prediction](#deep-neural-network-training-tuning-and-prediction)
- [Deep Neural Network](#deep-neural-network)
- [Week 2 Quiz](#week-2-quiz)
- [Week 2 Wrap Up](#week-2-wrap-up)
- [Lecture Notes Week 2](#lecture-notes-week-2)

Durante la semana 2, intensificamos nuestro enfoque en cómo preparar adecuadamente características y etiquetas, facilitando el camino hacia la construcción de una red neuronal de una sola capa. Exploraremos en detalle cómo alimentar conjuntos de datos segmentados a nuestras redes, optimizar el tamaño del lote, y ajustar la tasa de aprendizaje dinámicamente. La introducción de las Redes Neuronales Profundas (DNN) y su entrenamiento, sintonización y predicción ocupará un lugar central, proporcionándote una comprensión robusta de estos conceptos clave para el análisis de series temporales.

## A Conversation with Andrew Ng W2
[<- Return to INDEX 2](#index-2)

![img.png](ims%2FW2%2Fimg.png)

Bienvenidos a la segunda semana de este curso. Aprendió acerca de cómo calcular el análisis estadístico de una serie temporal. En esta semana aprenderás a aplicar una nueva red a estas secuencias. Sí, vamos a empezar con un DNN relativamente simple, y si recuerdas DNN todo el camino como al comienzo de la especialización, así que vas a ser capaz de llevar esas habilidades a tener en cuenta los datos de series de tiempo.

Así que vamos a construir un DNN muy simple al igual que un DNN de tres capas. Entonces una técnica, aunque es que me parece realmente útil que vas a aprender esta semana, es ser capaz de ajustar la tasa de aprendizaje del optimizador. Pasaremos mucho tiempo ajustando la tasa de aprendizaje.

![img_1.png](ims%2FW2%2Fimg_1.png)

Uno de mis recuerdos más distintivos de un par de décadas atrás fue un pasante de verano en investigación de Microsoft permaneciendo hasta 3:00 AM en la oficina, afinando las tasas de aprendizaje. Me gustaría poder ajustar mi propia tasa de aprendizaje a veces. Así que creo que dar a la gente una manera sistemática de hacer esto, será muy útil. Sí. Creo que sí, y sé que he pasado demasiadas noches sin dormir en ello. 

Así que tratamos de automatizar aquí y así enseñarte una técnica para automatizar y espero que sea útil, y espero que te ayude a empezar a aprender cómo predecir tus datos de series de tiempo. Al final de esta semana, usted ha entrenado DNN en los datos de la serie del tiempo que vio la semana pasada y se llega a ver lo bien que lo hace. Por favor, ve al siguiente video.

## Preparing Features and Labels Notebook (Lab 1)
[<- Return to INDEX 2](#index-2)

En los próximos vídeos, Laurence le mostrará una serie de comandos para preparar características y etiquetas. Puede utilizar el cuaderno de 
aquí [C4_W2_Lab_1_features_and_labels.ipynb](notebooks%2FW2%2FC4_W2_Lab_1_features_and_labels.ipynb)
 para seguirle la pista y ver los enlaces a la documentación de cada comando.

## Preparing Features and Labels
[<- Return to INDEX 2](#index-2)

![img_2.png](ims%2FW2%2Fimg_2.png)

La semana pasada buscó crear un conjunto de datos estacionales sintéticos que contenía tendencia, estacionalidad y un poco de ruido. También ha mirado algunos métodos estadísticos para analizar el conjunto de datos y hacer predicciones a partir de él. Algunos de los resultados que obtuviste fueron realmente bastante buenos, pero aún no se aplicó el aprendizaje automático. Esta semana, vas a ver el uso de algunos métodos de aprendizaje automático con los mismos datos. Veamos a dónde nos puede llevar el aprendizaje automático.

![img_3.png](ims%2FW2%2Fimg_3.png)

En primer lugar, como con cualquier otro problema ML, tenemos que dividir nuestros datos en características y etiquetas. En este caso nuestra característica es efectivamente una serie de valores en la serie, con nuestra etiqueta siendo el siguiente valor. Llamaremos a ese número de valores que tratarán como nuestra característica, el tamaño de la ventana, donde estamos tomando una ventana de los datos y entrenando un modelo ML para predecir el siguiente valor. Así que, por ejemplo, si tomamos nuestros datos de series temporales, digamos, 30 días a la vez, usaremos 30 valores como entidad y el siguiente valor será la etiqueta. Luego, con el tiempo, entrenaremos una red neuronal para que coincida con las 30 características con la única etiqueta.

![img_4.png](ims%2FW2%2Fimg_4.png)

Así que vamos, por ejemplo, a utilizar la clase TF.Data.Dataset para crear algunos datos para nosotros, vamos a hacer un rango de 10 valores.
Cuando los imprimiremos veremos una serie de datos del 0 al 9.

![img_5.png](ims%2FW2%2Fimg_5.png)

Así que ahora vamos a hacerlo un poco más interesante. Utilizaremos dataset.window para expandir nuestro conjunto de datos usando windowing.

Sus parámetros son el tamaño de la ventana y cuánto queremos cambiar por cada vez. Así que si establecemos un tamaño de ventana de 5 con un desplazamiento de 1 cuando lo imprimimos, veremos algo como esto, 01234, que simplemente se detiene allí porque son cinco valores, entonces vemos 12345 etc, etc. Una vez que lleguemos al final del conjunto de datos tendremos menos valores porque simplemente no existen. Así que obtendremos 6789, y luego 789, etc., etc.

![img_6.png](ims%2FW2%2Fimg_6.png)

Así que vamos a editar nuestra ventana un poco, para que tengamos datos de tamaño regular. Podemos hacerlo con un parámetro adicional en la ventana llamado drop_remainder. Y si ponemos esto en verdadero, truncará los datos dejando caer todos los restos. A saber, esto significa que sólo nos dará ventanas de cinco elementos. Así que cuando lo imprima, ahora se verá así, comenzando en 01234 y terminando en 56789.

![img_7.png](ims%2FW2%2Fimg_7.png)

Genial, ahora vamos a poner estos en numpy listas para que podamos empezar a usarlos con el aprendizaje automático. La buena noticia es, es que esto es súper fácil, simplemente llamamos al método .numpy en cada elemento en el conjunto de datos, y cuando imprimimos ahora vemos que tenemos una lista numpy.

![img_8.png](ims%2FW2%2Fimg_8.png)

De acuerdo, lo siguiente es dividir los datos en entidades y etiquetas. Para cada elemento de la lista, tiene sentido tener todos los valores excepto el último en ser la característica, y luego el último puede ser la etiqueta. Y esto se puede lograr con mapeo, así, donde nos dividimos en todo pero el último con: -1, y luego solo el último con -1:. Lo que nos da esta salida cuando imprimimos, que ahora parece un buen conjunto de características y etiquetas. 

![img_9.png](ims%2FW2%2Fimg_9.png)

Normalmente, usted barajaría sus datos antes de entrenar. Y esto es posible usando el método aleatorio. Lo llamamos con el tamaño del búfer de diez, porque esa es la cantidad de elementos de datos que tenemos.
Y cuando imprima los resultados, veremos que nuestras características y conjuntos de etiquetas se han barajado.

![img_10.png](ims%2FW2%2Fimg_10.png)

Finalmente, podemos ver el procesamiento por lotes de los datos, y esto se hace con el método por lotes.
Tomará un parámetro de tamaño, y en este caso es 2. Así que lo que haremos es agrupar los datos en conjuntos de dos, y si los imprimimos, veremos esto. Ahora tenemos tres lotes de dos elementos de datos cada uno. Y si nos fijamos en el primer set, verá las x e y correspondientes. Así que cuando x sea cuatro, cinco, seis y siete, nuestra y es ocho, o cuando x es cero, uno, dos, tres, verá que nuestra y es cuatro.


![img_11.png](ims%2FW2%2Fimg_11.png)

Bien, ahora que has visto las herramientas que nos permiten crear una serie de x e y, o características y etiquetas, tienes todo lo que necesitas para trabajar en un conjunto de datos con el fin de obtener predicciones de él. Vamos a echar un vistazo a un reparto de pantalla de este código, antes de pasar a crear nuestras primeras redes neuronales para ejecutar predicciones sobre estos datos.

## Preparing Features and Labels (screencast)
[<- Return to INDEX 2](#index-2)

En el video anterior, vio cómo preparar una serie temporal para el aprendizaje automático mediante la ventana de los datos. Viste el código para crear un conjunto de datos muy simple, y luego aprendiste cómo puedes preparar ese conjunto de datos como entidades y etiquetas o x e y. En esta clase, pasarás por un screencast de un cuaderno que contiene todo ese código.

> El código está en: [C4_W2_Lab_1_features_and_labels.ipynb](notebooks%2FW2%2FC4_W2_Lab_1_features_and_labels.ipynb)

El resumen es exactamente igual que la explicación anterior pero en un notebook.

## Single Layer Network Notebook (Lab 2)
[<- Return to INDEX 2](#index-2)

Las próximas clases mostrarán cómo puede alimentar conjuntos de datos con ventanas a una única red neuronal. Puede utilizar el cuaderno de notas 
aquí [C4_W2_Lab_2_single_layer_NN.ipynb](notebooks%2FW2%2FC4_W2_Lab_2_single_layer_NN.ipynb)
 para seguirnos o puede ejecutarlo después del screencast.

## Feeding Windowed Dataset into Neural Network
[<- Return to INDEX 2](#index-2)

![img_12.png](ims%2FW2%2Fimg_12.png)

En los últimos videos vio cómo preparar datos de series de tiempo para el aprendizaje automático creando un conjunto de datos de ventana donde los valores n anteriores podrían verse como las entidades de entrada son x. Y el valor actual con cualquier marca de tiempo es la etiqueta de salida o la y. 

![img_13.png](ims%2FW2%2Fimg_13.png)

Entonces se vería un poco como Esto Con una serie de valores de entrada en x, normalmente llamada ventana en los datos. Usted vio en los videos anteriores cómo usar las herramientas en datasets de flujo tensor para crear estas ventanas. Esta semana adaptarás ese código para alimentar una red neuronal y luego entrenarás en los datos.

![img_14.png](ims%2FW2%2Fimg_14.png)

Así que comencemos con esta función que llamará a un conjunto de datos de Windows. Tomará en una serie de datos junto con los parámetros para el tamaño de la ventana que queremos. El tamaño de los lotes que se utilizarán al entrenar, y el tamaño del búfer aleatorio, que determina cómo se barajarán los datos.

- El primer paso será crear un dataset a partir de la serie utilizando un dataset tf.data. Y le pasaremos la serie usando su método from_tensor_slices.
- A continuación, utilizaremos el método de ventana del conjunto de datos basado en nuestro window_size para cortar los datos en las ventanas apropiadas. Cada uno está siendo desplazado por un tiempo determinado. Los mantendremos todos del mismo tamaño estableciendo el resto de la gota en verdadero.
- A continuación, aplanamos los datos para facilitar el trabajo con ellos. Y será aplanado en trozos del tamaño de nuestra window_size + 1.

![img_15.png](ims%2FW2%2Fimg_15.png)

Una vez que está aplanado, es fácil barajar. Llamas a un shuffle y le pasas el búfer shuffle. El uso de un búfer aleatorio acelera las cosas un poco. Entonces, por ejemplo, si tiene 100.000 elementos en su conjunto de datos, pero establece el búfer en mil. Simplemente llenará el búfer con los primeros mil elementos, elige uno de ellos al azar. Y luego reemplazará eso con el primer elemento 1,000 y antes de elegir aleatoriamente de nuevo, y así sucesivamente. De esta manera, con conjuntos de datos súper grandes, el elemento aleatorio que elige puede elegir entre un número más pequeño que acelera efectivamente las cosas.

El conjunto de datos mezclados se divide entonces en xs, que es todos los elementos excepto el último, y el y que es el último elemento.

A continuación, se empila en el tamaño del lote seleccionado y se devuelve.

## Single Layer Neural Network
[<- Return to INDEX 2](#index-2)

![img_16.png](ims%2FW2%2Fimg_16.png)

Ahora que tenemos un conjunto de datos de ventana, podemos comenzar a entrenar redes neuronales con él. Comencemos con una super simple que es efectivamente una regresión lineal. Mediremos su precisión, y luego trabajaremos desde allí para mejorarlo. Antes de que podamos hacer un entrenamiento, tenemos que dividir nuestro conjunto de datos en conjuntos de entrenamiento y validación. 

![img_17.png](ims%2FW2%2Fimg_17.png)

Aquí está el código para hacer eso en el paso de tiempo 1000. Podemos ver que los datos de entrenamiento es el subconjunto de la serie llamada x tren hasta el tiempo de división. 

![img_18.png](ims%2FW2%2Fimg_18.png)

Aquí está el código para hacer una regresión lineal simple. Vamos a verlo línea por línea. Comenzaremos configurando todas las constantes que queremos pasar a la función de dataset de ventana. Estos incluyen el tamaño de la ventana en los datos, el tamaño del lote que queremos para el entrenamiento, y el tamaño del buffer barajado como acabamos de discutir. 

A continuación, vamos a crear nuestro conjunto de datos. Lo haremos tomando nuestra serie y en el cuaderno que pasarás más adelante, crearás la misma serie sintética que hiciste en la primera semana. Pasará su serie a lo largo de lo que su tamaño de ventana deseado, tamaño de lote, y tamaño de búfer barajado, y le devolverá un conjunto de datos formateados que podría usar para la capacitación. Luego voy a crear una sola capa densa con su forma de entrada es el tamaño de la ventana. Para la regresión lineal, eso es todo lo que necesita. 

Estoy usando este enfoque. Al pasar la capa a una variable llamada L0, porque más tarde quiero imprimir sus pesos aprendidos, y es mucho más fácil para mí hacerlo si tengo una variable para referirse a la capa para eso. Entonces simplemente defino mi modelo como una secuencia que contiene la capa única así. 

![img_19.png](ims%2FW2%2Fimg_19.png)

Ahora compilaré y ajustaré mi modelo con este código. Voy a utilizar la función de pérdida de error cuadrado media estableciendo la pérdida a MSE, y mi optimizador usará el Descenso de gradiente estocástico. Usaría esta metodología en lugar de la cadena cruda, para poder establecer parámetros para inicializarla como la tasa de aprendizaje o LR y el impulso. 

Experimente con diferentes valores aquí para ver si puede obtener su modelo para converger más rápido o con mayor precisión. A continuación, puede adaptarse a su modelo simplemente pasándole el conjunto de datos, que ya ha sido preformateado con los valores x e y.

Voy a correr por 100 épocas aquí. Ignorando la salida de epoch pero epoch estableciendo verbose en cero. Una vez que haya terminado el entrenamiento, puede realmente inspeccionar los diferentes pesos con este código. 

![img_20.png](ims%2FW2%2Fimg_20.png)

Recuerde antes cuando nos referimos a la capa con una variable llamada L 0? Bueno, aquí es donde eso es útil. La salida se verá así. Si lo inspecciona de cerca, verá que la primera matriz tiene 20 valores, y la secundaria tiene solo un valor. 

Esto se debe a que la red ha aprendido una regresión lineal para adaptarse a los valores lo mejor posible. Así que cada uno de los valores de la primera matriz puede ser visto como los pesos para los 20 valores en x, y el valor para la segunda matriz es el valor b.


## Machine Learning on Time Windows
[<- Return to INDEX 2](#index-2)

Entonces, si piensas en este diagrama y consideras que la ventana de entrada tiene 20 valores de ancho, , entonces llamémoslos x0, x1, x2, etc., hasta x19. Pero seamos claros. Ese no es el valor en el eje horizontal que comúnmente se llama el eje x, es el valor de la serie de tiempo en ese punto en el eje horizontal. 

![img_21.png](ims%2FW2%2Fimg_21.png)

 Así que el valor en el tiempo t0, que es 20 pasos antes de que el valor actual se llame x0, y t1 se llama x1, etc. Del mismo modo, para la salida, que entonces consideraríamos que es el valor en el momento actual es la y.

## Prediction
[<- Return to INDEX 2](#index-2)


![img_22.png](ims%2FW2%2Fimg_22.png)

Así que ahora, si miramos los valores de nuevo y vemos que estos son los pesos para los valores en esa marca de tiempo particular y b es el sesgo o la pendiente, 

![img_23.png](ims%2FW2%2Fimg_23.png)

podemos hacer una regresión lineal estándar así para predecir el valor de y en cualquier paso multiplicando los valores x por los pesos y luego agregar el sesgo. 

![img_24.png](ims%2FW2%2Fimg_24.png)

Así que, por ejemplo, si tomo 20 elementos de mi serie e imprimo, puedo ver los valores 20x. Si quiero predecirlas, puedo pasar esa serie a mi modelo para obtener una predicción. El nuevo eje NumPy luego simplemente lo cambia a la dimensión de entrada que es utilizada por el modelo. 

![img_25.png](ims%2FW2%2Fimg_25.png)

La salida se verá así. La matriz superior es los 20 valores que proporcionan la entrada a nuestro modelo y la parte inferior es el valor previsto de vuelta desde el modelo. 

Así que hemos entrenado a nuestro modelo para decir que cuando ve 20 valores como este, el siguiente valor previsto es 49.08478. Así que si queremos trazar nuestras previsiones para cada punto en la serie temporal relativa a los 20 puntos anteriores donde nuestro tamaño de ventana era 20, podemos escribir código como este. 

![img_26.png](ims%2FW2%2Fimg_26.png)

Creamos una lista vacía de pronósticos y luego iteramos sobre la serie tomando rebanadas y tamaño de ventana, prediciendo, y agregando los resultados a la lista de pronósticos.

Habíamos dividido nuestra serie de tiempo en entrenamiento y pruebas sentido tomando todo antes de un cierto tiempo es entrenamiento y el resto es validación. Así que tomaremos las previsiones después de el tiempo de división y las cargaremos en una matriz NuimPy para gráficos. 

![img_27.png](ims%2FW2%2Fimg_27.png)

Ese gráfico se ve así con los valores reales en azul y los predichos en naranja. Se puede ver que nuestras predicciones se ven bastante bueno y obtenerlas fue relativamente simple en comparación con toda la gimnasia estadística que tuvimos que hacer en los últimos videos. 

![img_28.png](ims%2FW2%2Fimg_28.png)

Así que vamos a medir el error absoluto medio como hemos hecho antes, y podemos ver que estamos en un estadio similar al que estábamos con un análisis complejo que hicimos anteriormente. 

![img_29.png](ims%2FW2%2Fimg_29.png)

Ahora eso es solo usar una sola capa en una red neuronal para calcular una regresión lineal. Veamos si podríamos hacerlo mejor con un DNN completamente conectado a continuación. Antes de llegar a eso, vamos a revisar el libro de trabajo para esta lección para asegurarnos de que entiendas todo lo que hemos hecho hasta ahora. El siguiente video será un screencast de pasar por eso y luego trabajarás en el DNN después de eso.

## More on Single Layer Neural Network
[<- Return to INDEX 2](#index-2)

![img_30.png](ims%2FW2%2Fimg_30.png)

En el último video, vio cómo escribir código para obtener un modelo de regresión lineal de aprendizaje automático de los datos de series de tiempo. En este video, te paso a través del cuaderno con todo el código y luego puedes probarlo por ti mismo.

> Nota: el notebook de esta lección está en -> [C4_W2_Lab_2_single_layer_NN.ipynb](notebooks%2FW2%2FC4_W2_Lab_2_single_layer_NN.ipynb)


## Deep Neural Network Notebook (Lab 3)
[<- Return to INDEX 2](#index-2)

La última sección de esta semana demostrará el uso de redes neuronales multicapa y puede utilizar el cuaderno de 
aquí [C4_W2_Lab_3_deep_NN.ipynb](notebooks%2FW2%2FC4_W2_Lab_3_deep_NN.ipynb)
 como referencia para los próximos vídeos.

## Deep Neural Network Training, Tuning and Prediction
[<- Return to INDEX 2](#index-2)

![img_31.png](ims%2FW2%2Fimg_31.png)

Vale, así que esta semana has estado buscando aprendizaje automático para predicando el siguiente valor en una secuencia. Usted ha aprendido cómo dividir sus datos en fragmentos de ventana en los que podría entrenar , y luego vio una simple red neuronal de una sola capa que le dio lo que efectivamente era una regresión lineal. Ahora vamos a tomar eso al siguiente paso con un DNN para ver si podemos mejorar la precisión de nuestro modelo. 

![img_32.png](ims%2FW2%2Fimg_32.png)

No es muy diferente del modelo de regresión lineal que vimos anteriormente. Y esta es una red neuronal profunda relativamente simple que tiene tres capas. Así que desempacémoslo línea por línea. Primero tendremos que obtener un conjunto de datos que generará pasando datos x_train, junto con el tamaño deseado de la ventana, el tamaño del lote y el tamaño del búfer aleatorio.

A continuación, definiremos el modelo. Vamos a mantenerlo simple con tres capas de 10, 10 y 1 neuronas. La forma de entrada es del tamaño de la ventana y activaremos cada capa usando un relu.

A continuación, compilaremos el modelo como antes con una función de pérdida de error cuadrado medio y optimizador de gradiente estocástico.

![img_33.png](ims%2FW2%2Fimg_33.png)

Finalmente, encajaremos el modelo en más de 100 épocas, y después de unos segundos de entrenamiento, veremos resultados que se ven así. 

![img_34.png](ims%2FW2%2Fimg_34.png)

Es bastante bueno todavía. Y cuando calculamos el error absoluto medio, somos más bajos que antes, , así que es un paso en la dirección correcta. Pero también es un poco una puñalada en la oscuridad, particularmente con la función optimizador. ¿No sería bueno si pudiéramos elegir la tasa de aprendizaje óptima en lugar de la que elegimos?


![img_35.png](ims%2FW2%2Fimg_35.png)

Podríamos aprender de manera más eficiente y construir un modelo mejor. Ahora veamos una técnica para que usa devoluciones de llamada que usó hace mucho tiempo en el primer curso. Así que aquí hay un código para la red neuronal anterior. Pero he agregado una devolución de llamada para ajustar la tasa de aprendizaje usando un programador de tasas de aprendizaje. Puedes ver ese código aquí.

![img_36.png](ims%2FW2%2Fimg_36.png)

Esto será llamado en la devolución de llamada al final de cada época. Lo que hará es cambiar las tasas de aprendizaje a un valor basado en el número de la época. Así que en la época 1, es 1 veces 10 a -8 veces 10 al poder de 1 sobre 20. Y para cuando lleguemos a la época 100, será 1 veces 10 a -8 veces 10 al poder de 5, y eso es 100 sobre 20.

![img_37.png](ims%2FW2%2Fimg_37.png)

Esto sucederá en cada devolución de llamada porque lo configuramos en el parámetro callbacks del atuendo modelado.

![img_38.png](ims%2FW2%2Fimg_38.png)

Después de entrenar con esto, podemos trazar la última por época contra la tasa de aprendizaje por época usando este código, y veremos un gráfico como este.

![img_39.png](ims%2FW2%2Fimg_39.png)

El eje y nos muestra la pérdida para esa época y el eje x nos muestra la tasa de aprendizaje. Entonces podemos tratar de elegir el punto más bajo de la curva donde todavía es relativamente estable así, y eso es justo alrededor de 7 veces 10 a -6.

![img_40.png](ims%2FW2%2Fimg_40.png)

Así que vamos a establecer que sea nuestra tasa de aprendizaje y luego volveremos a entrenar. Así que aquí está el mismo código de red neuronal, y hemos actualizado la tasa de aprendizaje, así que también lo entrenaremos un poco más. Vamos a comprobar los resultados después del entrenamiento para 500 épocas.

![img_41.png](ims%2FW2%2Fimg_41.png)

Aquí están los códigos para trazar la pérdida que se calculó durante el entrenamiento, y nos dará un gráfico como este.

![img_42.png](ims%2FW2%2Fimg_42.png)

Lo cual en la primera inspección parece que probablemente estamos perdiendo nuestro tiempo entrenando más allá de tal vez solo 10 épocas, pero está algo sesgado por el hecho de que las pérdidas anteriores eran tan altas. 

![img_43.png](ims%2FW2%2Fimg_43.png)

Si los recortamos y trazamos la pérdida para épocas después del número 10 con código como esto, entonces el gráfico nos contará una historia diferente.

![img_44.png](ims%2FW2%2Fimg_44.png)

Podemos ver que la pérdida continuaba disminuyendo incluso después de 500 épocas. Y eso demuestra que nuestra red está aprendiendo muy bien.

![img_45.png](ims%2FW2%2Fimg_45.png)

Y los resultados de las predicciones superpuestas contra los originales se ven así.

![img_46.png](ims%2FW2%2Fimg_46.png)

Y el error absoluto medio en los resultados es significativamente menor que antes.

![img_47.png](ims%2FW2%2Fimg_47.png)

Te llevaré a través de un screencast de este código en acción en el siguiente video. Usando un DNN muy simple, hemos mejorado nuestros resultados muy bien. Pero sigue siendo sólo un DNN, no hay secuenciación tomada en cuenta, y en una serie temporal como esta, los valores que están inmediatamente antes de un valor son más propensos a impactarlo que aquellos más allá en el pasado. Y esa es la configuración perfecta para usar RNS como teníamos en el curso en lenguaje natural. Ahora veremos eso la próxima semana, pero primero, vamos a profundizar en este código.

## Deep Neural Network
[<- Return to INDEX 2](#index-2)

![img_48.png](ims%2FW2%2Fimg_48.png)

En el video anterior, examinamos el entrenamiento de una red neuronal profunda para aprender, para predecir el siguiente valor en una secuencia de Windows. También analizamos algunos ajustes básicos de hiperparámetros para elegir una buena tasa de aprendizaje para el descenso del gradiente, lo que nos permitió mejorar aún más el modelo. En este video, vamos a terminar esta semana revisando el libro de trabajo que muestra cómo hacer todo eso.

> Nota el notebook de esta sección está en: [C4_W2_Lab_3_deep_NN.ipynb](notebooks%2FW2%2FC4_W2_Lab_3_deep_NN.ipynb)

![img_49.png](ims%2FW2%2Fimg_49.png)

Así que eso termina esta semana. Vaya a través del libro de trabajo usted mismo y experimente con diferentes definiciones de red neuronal , cambiando alrededor de las capas y cosas así para ver si podemos hacerlo aún mejor.

La semana que viene vamos a llevar esto al siguiente nivel usando los tipos de red neuronal que eran redes neuronales actuales que tienen capacidades de secuenciación incorporadas. Nos vemos allí.

## Week 2 Quiz
[<- Return to INDEX 2](#index-2)

1. **Question 1**
What is a windowed dataset?

   - [ ] The time series aligned to a fixed shape
   - [ ] A consistent set of subsets of a time series
   - [ ] There’s no such thing
   - [x] A fixed-size subset of a time series

   > **Correct!** A windowed dataset involves creating a fixed-size subset from a time series data for processing or model training purposes.

2. **Question 2**
What does ‘drop_remainder=True’ do?

   - [x] It ensures that all rows in the data window are the same length by cropping data
   - [ ] It ensures that all data is used
   - [ ] It ensures that all rows in the data window are the same length by adding data
   - [ ] It ensures that the data is all the same shape

   > **Correct!** Using 'drop_remainder=True' when preparing datasets ensures uniformity in data window size by cutting off any excess rows.

3. **Question 3**
What’s the correct line of code to split an n column window into n-1 columns for features and 1 column for a label

   - [ ] dataset = dataset.map(lambda window: (window[n-1], window[1]))
   - [x] dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
   - [ ] dataset = dataset.map(lambda window: (window[-1:], window[:-1]))
   - [ ] dataset = dataset.map(lambda window: (window[n], window[1]))

   > **Correct!** The code snippet `dataset = dataset.map(lambda window: (window[:-1], window[-1:]))` correctly splits the input window into features and labels.

4. **Question 4**
What does MSE stand for?

   - [ ] Mean Series error
   - [ ] Mean Second error
   - [x] Mean Squared error
   - [ ] Mean Slight error

   > **Correct!** MSE stands for Mean Squared Error, a common loss function used in regression problems to quantify the difference between the predicted and actual values.

5. **Question 5**
What does MAE stand for?

   - [ ] Mean Average Error
   - [ ] Mean Advanced Error
   - [x] Mean Absolute Error
   - [ ] Mean Active Error

   > **Correct!** MAE stands for Mean Absolute Error, another loss function used to measure how close predictions are to the outcomes.

6. **Question 6**
If time values are in time[], series values are in series[] and we want to split the series into training and validation at time split_time, what is the correct code?

   - [x] 
     ```python
     time_train = time[:split_time]
     x_train = series[:split_time]
     time_valid = time[split_time:]
     x_valid = series[split_time:]
     ```
   - [ ] Other options...

   > **Correct!** This code snippet correctly divides the dataset into training and validation sets according to the `split_time` index.

7. **Question 7**
If you want to inspect the learned parameters in a layer after training, what’s a good technique to use?

   - [ ] Run the model with unit data and inspect the output for that layer.
   - [x] Assign a variable to the layer and add it to the model using that variable. Inspect its properties after training.
   - [ ] Decompile the model and inspect the parameter set for that layer.
   - [ ] Iterate through the layers dataset of the model to find the layer you want.

   > **Correct!** Assigning a variable to a layer for later inspection is an effective method to understand learned parameters post-training.

8. **Question 8**
How do you set the learning rate of the SGD optimizer?

   - [ ] You can’t set it
   - [ ] Use the Rate property
   - [ ] Use the RateOfLearning property
   - [x] Use the learning_rate property

   > **Correct!** The `learning_rate` property is used to define the learning rate for the SGD (Stochastic Gradient Descent) optimizer.

9. **Question 9**
If you want to amend the learning rate of the optimizer on the fly, after each epoch. What do you do?

   - [ ] Use a LearningRateScheduler and pass it as a parameter to a callback
   - [ ] Callback to a custom function and change the SGD property
   - [x] Use a LearningRateScheduler object in the callbacks namespace and assign that to the callback
   - [ ] You can’t set it

   > **Correct!** Utilizing a LearningRateScheduler allows for dynamic adjustment of the learning rate throughout the training process.

## Week 2 Wrap Up
[<- Return to INDEX 2](#index-2)

Ya ha explorado los datos de series temporales y ha visto cómo dividirlos en conjuntos de entrenamiento y validación para entrenar una DNN.

Pero los datos de secuencias tienden a funcionar mejor con RNNs. Así que la semana que viene, va a ver cómo entrenar un modelo utilizando RNNs y LSTMs en estos datos para ver qué ocurre.

## Lecture Notes Week 2
[<- Return to INDEX 2](#index-2)

Los apuntes de las conferencias están disponibles aquí: [C4_W2.pdf](notes%2FC4_W2.pdf)

# Weekly Assignment Prediction with a DNN
[<- Return to INDEX 0](#index-0)

En clase ha visto cómo dividir un conjunto de datos y cómo empezar a entrenar una DNN utilizándolo. Para este ejercicio creará su propio conjunto de datos sintéticos.

Una vez que tenga su serie, ¡creará una DNN para predecir los valores de esa serie!

Esta tarea se calificará en función del MSE de sus predicciones. Al final de la tarea habrá presentado su modelo entrenado, que se utilizará para calcular la previsión. El pronóstico debe tener un MSE de 30 o menos para aprobar esta tarea.

> Respuesta: [C4W2_Assignment.ipynb](notebooks%2FW2%2Fassigment%202%2FC4W2_Assignment.ipynb)

![img.png](notebooks%2FW2%2Fassigment%202%2Fimg.png)

## Forecasting Using Neural Networks
[<- Return to INDEX 2](#index-2)

# Recurrent Neural Networks for Time Series
[<- Return to INDEX 0](#index-0)

## INDEX 3

- [A Conversation with Andrew Ng W3](#a-conversation-with-andrew-ng-w3)
- [Conceptual Overview](#conceptual-overview)
- [RNN Notebook (Lab 1)](#rnn-notebook-lab-1)
- [Shape of the Inputs to the RNN](#shape-of-the-inputs-to-the-rnn)
- [Outputting a Sequence](#outputting-a-sequence)
- [Lambda Layers](#lambda-layers)
- [Adjusting the Learning Rate Dynamically](#adjusting-the-learning-rate-dynamically)
- [More Info on Huber Loss](#more-info-on-huber-loss)
- [LSTM](#lstm)
- [Link to the LSTM Lesson](#link-to-the-lstm-lesson)
- [LSTM Notebook (Lab 2)](#lstm-notebook-lab-2)
- [Coding LSTMs](#coding-lstms)
- [Week 3 Quiz](#week-3-quiz)
- [Week 3 Wrap Up](#week-3-wrap-up)
- [Lecture Notes Week 3](#lecture-notes-week-3)

La tercera semana está dedicada a las Redes Neuronales Recurrentes (RNN), con un fuerte énfasis en entender las Formas de los Inputs hacia las RNN, la implementación de Capas Lambda, y el uso de LSTM (Long Short-Term Memory) para mejorar la precisión de nuestras predicciones. Examinaremos cómo las LSTMs pueden ser aplicadas a datos del mundo real, como los obtenidos de la observación de manchas solares, llevando a cabo ejercicios prácticos dirigidos a entrenar y afinar modelos para realizar predicciones precisas.

## A Conversation with Andrew Ng W3
[<- Return to INDEX 3](#index-3)

![img.png](ims%2FW3%2Fimg.png)

Bienvenidos a la tercera semana. En esta semana, puede aplicar RNN y LCM a los datos de secuencia de tiempo. La semana pasada aplicaste DNN a estos datos. Pero las series temporales son datos temporales, parece que deberías aplicar un modelo de secuencia , como un RNN o un LCM a eso. Así que eso es lo que haces esta semana.


![img_1.png](ims%2FW3%2Fimg_1.png)

En este extracto, se discute la aplicación de modelos de redes neuronales recurrentes (RNN) y modelos lineales de cadena (LCM) a datos de series de tiempo. Se compara el uso de estos modelos con el uso de redes neuronales profundas (DNN) en semanas anteriores, enfatizando que los modelos de secuencia como RNN y LCM son más adecuados para datos temporales debido a su capacidad para mantener el contexto a lo largo de una serie. Se menciona que los datos financieros, por ejemplo, pueden beneficiarse especialmente de estos modelos debido a la influencia temporal en los precios.

![img_2.png](ims%2FW3%2Fimg_2.png)

También se introduce el concepto de capas Lambda en Tensorflow y Keras, que permiten escribir código personalizado como parte de una red neuronal. Esto proporciona un mayor control sobre el preprocesamiento de datos dentro de la red neuronal misma, lo que puede simplificar el flujo de trabajo y brindar más flexibilidad al desarrollador.

## Conceptual Overview
[<- Return to INDEX 3](#index-3)

![img_3.png](ims%2FW3%2Fimg_3.png)

En las últimas dos semanas, has buscado crear redes neuronales para pronosticar datos de series temporales. Comenzaste con algunas técnicas analíticas simples, que luego lo extendiste a usar Machine Learning para hacer una regresión simple. Desde allí usas el DNN que ajustaste un poco para obtener un modelo aún mejor.

Esta semana, vamos a ver RNN para la tarea de predicción. Una red neuronal recurrente, o RNN es una red neuronal que contiene capas recurrentes. 

![img_4.png](ims%2FW3%2Fimg_4.png)

Estos están diseñados para procesos secuencialmente secuencia de entradas. RNN son bastante flexibles, capaces de procesar todo tipo de secuencias. Como viste en el curso anterior, podrían haber sido utilizados para predecir texto. Aquí los usaremos para procesar la serie temporal. 

Este ejemplo, creará un RNN que contiene dos capas recurrentes y una capa densa final, que servirá como salida. Con un RNN, puedes alimentarlo en lotes de secuencias, y emitirá un lote de pronósticos, al igual que hicimos la semana pasada. 

![img_5.png](ims%2FW3%2Fimg_5.png)

Una diferencia será que la forma de entrada completa cuando usando RNN es tridimensional. La primera dimensión será el tamaño del lote, la segunda será las marcas de tiempo, y la tercera es la dimensionalidad de las entradas en cada paso de tiempo. Por ejemplo, si es una serie temporal univariada, este valor será uno, para multivariante será más. 

Los modelos que has estado usando a fecha tenían entradas bidimensionales, la dimensión del lote fue la primera, y la segunda tenía todas las características de entrada. 
Pero antes de ir más lejos, vamos a profundizar en las capas RNN para ver cómo funcionan. 

![img_6.png](ims%2FW3%2Fimg_6.png)

Lo que parece que hay muchas celdas, en realidad solo hay una, y se usa repetidamente para calcular las salidas. En este diagrama, parece que hay muchos de ellos, pero estoy usando el mismo siendo reutilizado varias veces por la capa. En cada paso de tiempo, la celda de memoria toma el valor de entrada para ese paso. Así que, por ejemplo, es cero en el tiempo cero, y entrada de estado cero. 

![img_7.png](ims%2FW3%2Fimg_7.png)

Luego calcula la salida para ese paso, en este caso Y0, y un vector H0 de estado que se alimenta en el siguiente paso. H0 se introduce en la celda con X1 para producir Y1 y H1, que luego se introduce en la celda en el siguiente paso con X2 para producir Y2 y H2. 

![img_8.png](ims%2FW3%2Fimg_8.png)

Estos pasos continuarán hasta que lleguemos al final de nuestra dimensión de entrada, que en este caso tiene 30 valores. Ahora, esto es lo que le da a este tipo de arquitectura el nombre de una red neuronal recurrente, porque los valores se repiten debido a la salida de la celda, un paso que se devuelve en sí mismo en el siguiente paso de tiempo.

![img_9.png](ims%2FW3%2Fimg_9.png)

Como vimos en el curso de la PNL, esto es realmente útil para determinar estados. La ubicación de una palabra en una oración puede determinarla semántica. Del mismo modo, para series numéricas, cosas como números más cercanos en la serie podrían tener un impacto mayor que aquellos más lejos de nuestro valor objetivo.

## RNN Notebook (Lab 1)
[<- Return to INDEX 3](#index-3)

En los próximos vídeos, Laurence demostrará el uso de RNNs para construir su modelo predictivo. Puede utilizar el cuaderno 
aquí [C4_W3_Lab_1_RNN.ipynb](notebooks%2FW3%2FC4_W3_Lab_1_RNN.ipynb)
 para seguirle la pista.

## Shape of the Inputs to the RNN
[<- Return to INDEX 3](#index-3)

![img_10.png](ims%2FW3%2Fimg_10.png)

Vale. Hemos mencionado la forma de los datos y los lotes en los que se dividen los datos. Es importante echar un vistazo a eso, y vamos a profundizar en eso a continuación. Las entradas son tridimensionales. 


![img_11.png](ims%2FW3%2Fimg_11.png)

Así que, por ejemplo, si tenemos un tamaño de ventana de 30 marcas de tiempo y las estamos agrupando en tamaños de cuatro, la forma será 4 veces 30 veces 1, y cada marca de tiempo, la entrada de la celda de memoria será una matriz de cuatro por uno, como esta.


![img_12.png](ims%2FW3%2Fimg_12.png)

La celda también tomará la entrada de la matriz de estado del paso anterior. Pero, por supuesto, en este caso, en el primer paso, esto será cero. Para los siguientes, será la salida de la celda de memoria. 

![img_13.png](ims%2FW3%2Fimg_13.png)

Pero aparte del vector de estado, la celda, por supuesto, producirá un valor Y, que podemos ver aquí. Si la célula de memoria está compuesta por tres neuronas, entonces la matriz de salida será de cuatro por tres porque el tamaño del lote que llegó fue cuatro y el número de neuronas es tres. Así que la salida completa de la capa es tridimensional, en este caso, 4 por 30 por 3.

Con cuatro siendo el tamaño del lote, tres siendo el número de unidades, y 30 siendo el número total de pasos. En un RNN simple, la salida de estado H es solo una copia de la matriz de salida Y. Así que, por ejemplo, H_0 es una copia de Y_0, H_1 es una copia de Y_1, y así sucesivamente. 

![img_14.png](ims%2FW3%2Fimg_14.png)

Así que en cada marca de tiempo, la celda de memoria obtiene tanto la entrada actual como la salida anterior. Ahora, en algunos casos, es posible que desee ingresar una secuencia, pero no desea generar y solo desea obtener un único vector para cada instancia en el lote. Esto se suele llamar una secuencia al vector RNN. 

Pero en realidad, todo lo que haces es ignorar todas las salidas, excepto la última. Al usar Keras en TensorFlow, este es el comportamiento predeterminado. Así que si desea que la capa recurrente genere una secuencia, debe especificar secuencias de retorno igual a true al crear la capa. Tendrá que hacer esto cuando apile una capa RNN encima de otra.

## Outputting a Sequence
[<- Return to INDEX 3](#index-3)

![img_15.png](ims%2FW3%2Fimg_15.png)

Así que considere este RNN, estas tienen dos capas recuperadas, y la primera tiene return_sequences=true set up. Se producirá una secuencia que se alimenta a la siguiente capa. La siguiente capa no tiene return_sequence que se establece en True, por lo que solo dará salida al paso final. Pero observe la input_shape, está establecida en Ninguno y 1. 


ensorFlow asume que la primera dimensión es el tamaño del lote, y que puede tener cualquier tamaño en absoluto, por lo que no es necesario definirla. Entonces la siguiente dimensión es el número de marcas de tiempo, que podemos establecer en none, lo que significa que el RNN puede manejar secuencias de cualquier longitud. La última dimensión es sólo una porque estamos utilizando una unidad varía de series de tiempo. 

![img_16.png](ims%2FW3%2Fimg_16.png)


Si establecemos return_sequence en true y todas las capas recurrentes, entonces todas las secuencias de salida y la capa densa obtendrá una secuencia como sus entradas. Keras maneja esto usando la misma capa densa de forma independiente en cada sello de tiempo.

Puede parecer varios aquí, pero es el mismo que se está reutilizando en cada paso de tiempo. Esto nos da lo que se llama una secuencia para secuencia RNN. Se alimenta un lote de secuencias y devuelve un lote de secuencias de la misma longitud. La dimensionalidad puede no coincidir siempre. Depende del número de unidades en la venta de memoria. 


![img_15.png](ims%2FW3%2Fimg_15.png)

Así que volvamos ahora a un RNN de dos capas que tiene el segundo no secuencias de retorno. Esto nos dará una salida a un solo denso.

## Lambda Layers
[<- Return to INDEX 3](#index-3)

![img_17.png](ims%2FW3%2Fimg_17.png)

Pero me gustaría agregar un par de capas nuevas a esto, capas que usan el tipo Lambda. Este tipo de capa es una que nos permite realizar operaciones arbitrarias para efectivamente ampliar la funcionalidad de los kares de TensorFlow, y podemos hacerlo dentro de la propia definición del modelo. 


![img_18.png](ims%2FW3%2Fimg_18.png)

Así que la primera capa de Lambda será usada para ayudarnos con nuestra dimensionalidad. Si recuerda cuando escribimos la función auxiliar del dataset de ventana, devolvió lotes bidimensionales de Windows en los datos, con la primera siendo el tamaño del lote y la segunda el número de marcas de tiempo. Pero un RNN espera tres dimensiones; tamaño del lote, el número de marcas de tiempo, y la dimensionalidad de la serie. 

Con la capa Lambda, podemos solucionar esto sin reescribir nuestra función auxiliar de dataset de ventana. Usando el Lambda, simplemente expandemos la matriz por una dimensión. Al establecer la forma de entrada en none, estamos diciendo que el modelo puede tomar secuencias de cualquier longitud. 


![img_19.png](ims%2FW3%2Fimg_19.png)

 Del mismo modo, si ampliamos las salidas en 100, podemos ayudar a entrenar. La función de activación predeterminada en las capas RNN es tan H que es la activación tangente hiperbólica. Esto genera valores entre uno negativo y uno. Dado que los valores de las series de tiempo están en ese orden generalmente en los 10s como los 40s, 50s, 60s y 70s, entonces escalando las salidas al mismo estadio pueden ayudarnos con el aprendizaje.

Podemos hacer eso en una capa de Lambda también, simplemente lo multiplicamos por 100. Así que ahora echemos un vistazo a lo que se necesita para construir el RNN completo para que podamos empezar a hacer algunas predicciones con él. Lo verás en el siguiente video.

## Adjusting the Learning Rate Dynamically
[<- Return to INDEX 3](#index-3)

![img_25.png](ims%2FW3%2Fimg_25.png)

En el video anterior, usted consiguió un vistazo a RNN y cómo pueden ser utilizados para secuencia a vector a secuencia a secuencia predicción. Echemos un vistazo a codificándolos para el problema que nos ocupa y veamos si podemos obtener buenas predicciones en nuestras series de tiempo utilizándolas. 

Una cosa que verás en el resto de las lecciones en adelante es que me gustaría escribir un poco de código para optimizar la red neuronal para la tasa de aprendizaje del optimizador. Puede ser bastante rápido para entrenar y podemos desde allí ahorrar mucho tiempo en nuestro ajuste de hiperparámetros.

![img_26.png](ims%2FW3%2Fimg_26.png)

Así que aquí está el código para entrenar el RNN con dos capas cada una con 40 celdas. Para ajustar la tasa de aprendizaje, vamos a configurar una devolución de llamada, que se puede ver aquí. Cada época esto sólo cambia la tasa de aprendizaje un poco para que los pasos todo el camino de 1 veces 10 a el menos 8 a 1 veces 10 a menos 6. 

![img_27.png](ims%2FW3%2Fimg_27.png)

Puedes ver esa configuración aquí mientras entrena. También he introducido una nueva función de pérdida al uso llamado Huber que se puede ver aquí. La función Huber es una función de pérdida que es menos sensible a los valores atípicos y como estos datos pueden ser un poco ruidosos, vale la pena darle una oportunidad.

![img_28.png](ims%2FW3%2Fimg_28.png)

Si ejecuto esto por 100 épocas y mide la pérdida en cada época, Veré que mi tasa óptima de aprendizaje para gradiente estocástico desciende entre aproximadamente 10 a menos 5 y 10 a menos 6. 

![img_29.png](ims%2FW3%2Fimg_29.png)

Así que voy a establecer que es 5 veces 10 al menos 5. Así que ahora, voy a establecer mis modelos compilados con esa tasa de aprendizaje y el optimizador de descenso de gradiente estocástico.

![img_30.png](ims%2FW3%2Fimg_30.png)

Después de entrenar para 500 épocas, obtendré este gráfico, con un MAE en el conjunto de validación de aproximadamente 6.35. No está mal, pero me pregunto si podemos hacerlo mejor.

![img_31.png](ims%2FW3%2Fimg_31.png)

Así que aquí está la pérdida y el MAE durante el entrenamiento con el gráfico a la derecha es zoom en las últimas épocas. Como pueden ver, la tendencia fue genuinamente hacia abajo hasta un poco después de 400 épocas, cuando comenzó a ponerse inestable. 

![img_32.png](ims%2FW3%2Fimg_32.png)

Teniendo en cuenta esto, probablemente valga la pena solo entrenamiento para alrededor de 400 épocas. Cuando hago eso, obtengo estos resultados. Eso es más o menos lo mismo con el MAE sólo un poco más alto, pero hemos ahorrado 100 épocas que valen de entrenamiento para conseguirlo. Así que vale la pena. 

![img_33.png](ims%2FW3%2Fimg_33.png)

Un vistazo rápido a la formación MAE y la pérdida nos da estos resultados. Así que lo hemos hecho bastante bien, y eso solo estaba usando un simple RNN. Veamos cómo podemos mejorar esto con LSTM y lo verás en el siguiente video.

## More Info on Huber Loss
[<- Return to INDEX 3](#index-3)

Encontrará la página de Wikipedia 
[aquí](https://en.wikipedia.org/wiki/Huber_loss)
.

> Huber loss es una función de pérdida utilizada comúnmente en problemas de regresión en el aprendizaje automático y la estadística. Es una alternativa robusta a la pérdida cuadrática (también conocida como pérdida de error cuadrático medio o MSE) que puede ser sensible a valores atípicos en los datos.
> 

La función de pérdida de Huber está diseñada para ser menos sensible a los valores atípicos que la pérdida cuadrática. En lugar de penalizar los errores al cuadrado, el Huber loss penaliza los errores lineales para valores pequeños y errores cuadráticos para valores grandes. Esto lo hace adecuado para problemas en los que los datos pueden contener valores atípicos que afectan significativamente el rendimiento del modelo.



## LSTM
[<- Return to INDEX 3](#index-3)

![img_20.png](ims%2FW3%2Fimg_20.png)

En los vídeos anteriores, experimentaste con usando RNN para predecir valores en tu secuencia. Los resultados fueron buenos, pero necesitaron un poco de mejora ya que estabas golpeando extrañas mesetas en tus predicciones. Experimentaste con el uso de diferentes hiperparámetros y viste alguna mejora, pero tal vez un mejor enfoque sería usar LSTM en lugar de RNN para ver el impacto. 

![img_21.png](ims%2FW3%2Fimg_21.png)

Exploraremos eso en este video. Si recuerdas cuando mirabas RNN, se veían un poco así. Tenían celdas que tomaban parches como entradas o X, y calculaban una salida Y , así como el vector de estado, que alimentaba a la celda junto con la siguiente X que luego resultó en la Y, y el vector de estado y así sucesivamente. 



![img_22.png](ims%2FW3%2Fimg_22.png)

El impacto de esto es que mientras que el estado es un factor en los cálculos posteriores, sus impactos pueden disminuir considerablemente con las marcas de tiempo. LSTM son el estado celular a esto que mantiene un estado a lo largo la vida del entrenamiento para que el estado se pasa de celda a celda, marca de tiempo a marca de tiempo, y se puede mantener mejor.

![img_23.png](ims%2FW3%2Fimg_23.png)

Esto significa que los datos de anteriores en la ventana pueden tener un mayor impacto en la proyección general que en el caso de RNN. El estado también puede ser bidireccional para que el estado se mueva hacia adelante y hacia atrás. En el caso de los textos, esto fue realmente poderoso. 

![img_24.png](ims%2FW3%2Fimg_24.png)

Dentro de la predicción de secuencias numéricas, puede o no ser, y será ser interesante experimentar con. No voy a entrar en muchos detalles aquí, pero cientos de videos alrededor de LSTM son geniales. A partir de ahí, realmente se puede entender cómo funcionan bajo el capó.

## Link to the LSTM Lesson
[<- Return to INDEX 3](#index-3)

[Aquí](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)
 encontrará el enlace a la lección LSTM

## LSTM Notebook (Lab 2)
[<- Return to INDEX 3](#index-3)

Puede utilizar el cuaderno 
aquí [C4_W3_Lab_2_LSTM.ipynb](notebooks%2FW3%2FC4_W3_Lab_2_LSTM.ipynb)
 para seguir la demostración de codificación LSTM en el siguiente vídeo.

## Coding LSTMs
[<- Return to INDEX 3](#index-3)

![img_34.png](ims%2FW3%2Fimg_34.png)

Así que ahora echemos un vistazo a algún código para LSTM y cómo pueden funcionar con los datos con los que hemos estado jugando toda la semana. Así que aquí está la actualización de nuestro código para usar LSTM. Desempacemos y echemos un vistazo a las partes interesantes.

![img_36.png](ims%2FW3%2Fimg_36.png)

En primer lugar es tf.keras.backend.clear_session, y esto borra cualquier variable interna. Eso hace que sea fácil para nosotros experimentar sin modelos que impacten versiones posteriores de sí mismos.

![img_37.png](ims%2FW3%2Fimg_37.png)

Después de la capa Lambda que expande las dimensiones para nosotros he agregado una sola capa LSTM con 32 celdas. También he hecho un bidireccional para ver el impacto de eso en una predicción. La neurona de salida nos dará nuestro valor de predicción.

![img_38.png](ims%2FW3%2Fimg_38.png)

También estoy usando una tasa de aprendizaje de una vez 10 _ menos seis y eso podría valer la pena experimentar también. 

![img_39.png](ims%2FW3%2Fimg_39.png)

Así que aquí están los resultados de ejecutar este LSTM en los datos sintéticos que hemos estado usando durante todo el curso. La meseta bajo el gran pico es todavía allí y son MAE está en los bajos seises. No está mal, no es genial, pero no está mal.

![img_40.png](ims%2FW3%2Fimg_40.png)

Las predicciones parecen que podría haber un poco en el lado bajo también. Así que vamos a editar nuestro código para agregar otro LSTM para ver el impacto. Ahora puedes ver la segunda capa y tener en cuenta que tuvimos que establecer secuencias de retorno iguales a true en la primera para que esto funcione. Entrenamos en esto y ahora veremos los siguientes resultados. 

![img_41.png](ims%2FW3%2Fimg_41.png)

Aquí está el gráfico. Ahora está rastreando mucho mejor y más cerca de los datos originales. Tal vez no mantenerse al día con el fuerte aumento, pero al menos está rastreando cerca. También nos da un error promedio promedio que es mucho mejor y está mostrando que estamos yendo en la dirección correcta. 

![img_42.png](ims%2FW3%2Fimg_42.png)

Si editamos nuestra fuente para agregar una tercera capa LSTM como esta, agregando la capa y teniendo las secuencias de retorno de la segunda capa es true podemos luego entrenarla y ejecutarla, y obtendremos la siguiente salida. 

![img_43.png](ims%2FW3%2Fimg_43.png)

Realmente no hay mucho de una diferencia y son MAE realmente ha caído. Así que eso es todo para mirar LSTM y predecir nuestras secuencias. En el siguiente video echarás un vistazo a usando convoluciones para ver qué tendrá el impacto en usar para predecir contenido de series de tiempo.


## Week 3 Quiz
[<- Return to INDEX 3](#index-3)

1. **Question 1**
If X is the standard notation for the input to an RNN, what are the standard notations for the outputs?

   - [ ] Y
   - [ ] H
   - [x] Y(hat) and H
   - [ ] H(hat) and Y

   > **Correct!** In the context of RNNs, Y(hat) and H denote the standard notations for the outputs, representing the predicted value and hidden state, respectively.

2. **Question 2**
What is a sequence to vector if an RNN has 30 cells numbered 0 to 29?

   - [ ] The total Y(hat) for all cells
   - [x] The Y(hat) for the last cell
   - [ ] The Y(hat) for the second cell
   - [ ] The average Y(hat) for all 30 cells

   > **Correct!** A sequence to vector operation in an RNN refers to using the Y(hat) output from the last cell as the representative output for the entire sequence.

3. **Question 3**
What does a Lambda layer in a neural network do?

   - [ ] There are no Lambda layers in a neural network
   - [x] Allows you to execute arbitrary code while training
   - [ ] Changes the shape of the input or output data
   - [ ] Pauses training without a callback

   > **Correct!** A Lambda layer in a neural network allows for the execution of arbitrary, user-defined code, enabling custom operations that are not covered by standard layers.

4. **Question 4**
What does the axis parameter of tf.expand_dims do?

   - [x] Defines the dimension index at which you will expand the shape of the tensor 
   - [ ] Defines the axis around which to expand the dimensions
   - [ ] Defines the dimension index to remove when you expand the tensor
   - [ ] Defines if the tensor is X or Y

   > **Correct!** The axis parameter of tf.expand_dims specifies the index at which a new dimension will be added to the tensor, thereby expanding its shape.

5. **Question 5**
A new loss function was introduced in this module, named after a famous statistician. What is it called?

   - [x] Huber loss
   - [ ] Hawking loss
   - [ ] Hyatt loss
   - [ ] Hubble loss

   > **Correct!** The Huber loss, named after the statistician Peter J. Huber, is utilized for robust regression and combines the best properties of L1 and L2 loss functions.

6. **Question 6**
What’s the primary difference between a simple RNN and an LSTM?

   - [x] In addition to the H output, LSTMs have a cell state that runs across all cells 
   - [ ] LSTMs have multiple outputs, RNNs have a single one
   - [ ] In addition to the H output, RNNs have a cell state that runs across all cells 
   - [ ] LSTMs have a single output, RNNs have multiple

   > **Correct!** The key distinction between a simple RNN and an LSTM is that the latter includes both the H output and a cell state, enhancing memory and addressing vanishing gradient issues.

7. **Question 7**
If you want to clear out all temporary variables that TensorFlow might have from previous sessions, what code do you run?

   - [ ] tf.keras.clear_session
   - [ ] tf.cache.backend.clear_session()
   - [x] tf.keras.backend.clear_session()  
   - [ ] tf.cache.clear_session()

   > **Correct!** To reset the state of TensorFlow and clear out temporary variables from prior sessions, the `tf.keras.backend.clear_session()` function is used.

8. **Question 8**
What happens if you define a neural network with these two layers?
    ```python
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    
    tf.keras.layers.Dense(1),
    ```


   - [ ] Your model will fail because you need return_sequences=True after each LSTM layer
   - [ ] Your model will fail because you have the same number of cells in each LSTM
   - [x] Your model will fail because you need return_sequences=True after the first LSTM layer
   - [ ] Your model will compile and run correctly

   > **Correct!** When stacking LSTM layers in a sequence, it’s crucial to set `return_sequences=True` for all but the last LSTM layer to ensure the subsequent layer receives input sequences of the correct shape.


## Week 3 Wrap Up
[<- Return to INDEX 3](#index-3)

Ahora que ha desarrollado sus DNN con RNN y LSTM, es hora de ponerlo todo junto utilizando CNN y algunos datos del mundo real.

La semana que viene lo hará, ¡explorando la predicción de datos de manchas solares a partir de un par de cientos de años de mediciones!

## Lecture Notes Week 3
[<- Return to INDEX 3](#index-3)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C4_W3.pdf](notes%2FC4_W3.pdf)
.

# Weekly Assignment Using Layers for Sequence Processing
[<- Return to INDEX 0](#index-0)

## Forecast Using RNNs or LSTMs
[<- Return to INDEX 3](#index-3)

En clase ha aprendido sobre las RNN y las LSTM para la predicción, así como una metodología sencilla para elegir una tasa de aprendizaje decente para el optimizador de descenso de gradiente estocástico. 

En este ejercicio tomará un conjunto de datos sintéticos y escribirá el código para calcular predicciones utilizando capas como las RNNs o las LSTMs.

Esta tarea se calificará en función del MSE de sus previsiones. Al final de la tarea habrá entregado su modelo entrenado, que se utilizará para calcular la previsión. El pronóstico debe tener un MSE de 30 o menos para aprobar esta tarea.

> Respuesta: [C4W3_Assignment.ipynb](notebooks%2FW3%2Fassigment%203%2FC4W3_Assignment.ipynb)
> 

![img.png](notebooks%2FW3%2Fassigment%203%2Fimg.png)


# Real-world Time Series Data
[<- Return to INDEX 0](#index-0)

## INDEX 4

- [A Conversation with Andrew Ng W4](#a-conversation-with-andrew-ng-w4)
- [Convolutions](#convolutions)
- [Convolutional Neural Network Course](#convolutional-neural-network-course)
- [Bi-directional LSTMs](#bi-directional-lstms)
- [More on Batch Sizing](#more-on-batch-sizing)
- [Convolutions with LSTM (Lab 1)](#convolutions-with-lstm-lab-1)
- [Convolutions with LSTM](#convolutions-with-lstm)
- [Real Data - Sunspots](#real-data---sunspots)
- [Train and Tune the Model](#train-and-tune-the-model)
- [Prediction](#prediction)
- [Sunspots Notebooks (Lab 2 and Lab 3)](#sunspots-notebooks-lab-2-and-lab-3)
- [Sunspots](#sunspots)
- [Combining our Tools for Analysis](#combining-our-tools-for-analysis)
- [Week 4 Quiz](#week-4-quiz)
- [Lecture Notes Week 4](#lecture-notes-week-4)
- [Reminder About the End of Access to Lab Notebooks](#reminder-about-the-end-of-access-to-lab-notebooks)
- [Wrap Up](#wrap-up)
- [Congratulations](#congratulations)
- [References](#references)
- [Acknowledgments](#acknowledgments)
- [Specialization wrap up A conversation with Andrew Ng](#specialization-wrap-up-a-conversation-with-andrew-ng)
- [What next?](#what-next)
- [Opportunity to Mentor Other Learners](#opportunity-to-mentor-other-learners)

En la última semana, nos sumergiremos en la potente combinación de Convoluciones y Redes Neuronales, incluyendo la implementación de LSTMs bidireccionales y la integración de convoluciones con LSTMs. Mediante el estudio de datos reales sobre manchas solares, aprenderemos a entrenar y ajustar modelos para realizar predicciones, utilizando todas las herramientas y técnicas analizadas hasta este punto. Cerramos con una reflexión sobre todo lo aprendido, con énfasis en cómo aplicar este conocimiento en el desarrollo de futuros proyectos, y se presenta la oportunidad de convertirse en mentor de otros aprendices, aprovechando la experiencia adquirida.

## A Conversation with Andrew Ng W4
[<- Return to INDEX 4](#index-4)

![img.png](ims%2FW4%2Fimg.png)

La última semana de un curso de especialización se presenta como emocionante, con la oportunidad de aplicar un convnet 1D a datos reales de manchas solares. La especialización, llamada Flujo de Tensor en la práctica, busca integrar todo el aprendizaje en la práctica. Se está pasando de datos sintéticos a datos reales de manchas solares, lo que permitirá utilizar diversas técnicas como redes neuronales convolucionales, cámaras 1D, LSTM y redes neuronales profundas para predecir la actividad de las manchas solares. 

![img_1.png](ims%2FW4%2Fimg_1.png)

Al final de la semana, se espera que los estudiantes hayan construido un modelo sofisticado de red neuronal que integre estas técnicas. Además, se anima a los estudiantes a explorar cómo pueden aplicar lo aprendido en Tensor Flow para construir otras redes neuronales.

## Convolutions
[<- Return to INDEX 4](#index-4)

![img_2.png](ims%2FW4%2Fimg_2.png)

La semana pasada, revisamos las redes neuronales recurrentes incluyendo un simple RNN y un LSTM.
Usted vio cómo podrían ser útiles en el aprendizaje de una secuencia como la que hemos estado viendo. Y cómo los LSTM eliminaron algunos de los problemas que tuvimos con un RNN.

Esta semana, irás un paso más allá, combinando convoluciones con LSTM para obtener un modelo muy apropiado. Luego aplicaremos eso a los datos del mundo real en lugar de este conjunto de datos sintéticos en el que hemos estado trabajando desde el comienzo de este curso.


![img_3.png](ims%2FW4%2Fimg_3.png)

Aquí está el LSTM que miró la semana pasada, excepto que he agregado algo al comienzo de la pila secuencial.

Es un convo D donde intentaremos aprender 32 filtros. Es una convolución unidimensional. Así que tomaremos una ventana de cinco números y multiplicaremos los valores en esa ventana por los valores de filtro, de la misma manera que se hacen las convoluciones de imagen.

![img_4.png](ims%2FW4%2Fimg_4.png)


Vuelva al curso de circunvoluciones para obtener más información sobre eso.

## Convolutional Neural Network Course
[<- Return to INDEX 4](#index-4)

Encontrará más información sobre las CNN en el curso de Andrews dentro de la Especialización en Aprendizaje Profundo.  

Si siente curiosidad por el contenido y desea obtener más información, aquí tiene el 
[enlace](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
 al curso.

## Bi-directional LSTMs
[<- Return to INDEX 4](#index-4)

![img_5.png](ims%2FW4%2Fimg_5.png)

Una nota importante es que mientras nos deshacemos de la capa Lambda que reformó la entrada para que trabajemos con los LSTM. Así que en realidad estamos especificando una forma de entrada en la curva 1D aquí.

![img_6.png](ims%2FW4%2Fimg_6.png)

Esto requiere que actualicemos la función helper windowed_datasetet con la que hemos estado trabajando todo el tiempo. Simplemente usaremos tf.expand_ dims en la función auxiliar para expandir las dimensiones de la serie antes de procesarla. 

![img_7.png](ims%2FW4%2Fimg_7.png)

También similar a la semana pasada, el código intentará muchos diferentes tasas de aprendizaje cambiándolas época por época y trazando los resultados. Con estos datos y la red convolucional y basada en LAST, obtendremos una trama como esta. 

![img_8.png](ims%2FW4%2Fimg_8.png)

Claramente los fondos están alrededor de 10 al menos cinco después de lo cual parece un poco inestable, así que tomaremos eso como nuestras tasas de aprendizaje deseadas. Así, cuando definamos el optimizador establecerá la tasa de aprendizaje para ser 1e-5 como se muestra aquí. 

![img_9.png](ims%2FW4%2Fimg_9.png)

Cuando entrenemos para 500 épocas obtendremos esta curva. Es una gran mejora con respecto a anteriores. El pico ha perdido su meseta pero todavía no es del todo correcto, no está llegando lo suficientemente alto en relación con los datos. Ahora, por supuesto, el ruido es un factor y podemos ver locos fluctuaciones en el pico causado por el ruido, pero creo que nuestro modelo podría posiblemente hacer un poco mejor que esto.


![img_10.png](ims%2FW4%2Fimg_10.png)

Nuestro MAE está por debajo de cinco, pero apuesto a que fuera de ese primer pico es probablemente mucho más bajo que eso. Una solución podría ser entrenar un poco más. A pesar de que nuestras curvas de pérdida de MAE se ven planas en 500 épocas, podemos ver cuando hacemos zoom en que están disminuyendo lentamente. La red sigue aprendiendo aunque lentamente.

![img_11.png](ims%2FW4%2Fimg_11.png)

Ahora un método sería hacer tus LastMS bidireccional como este. Al entrenar, esto se ve muy bien dando muy baja pérdida en los valores de MAE a veces incluso menos de uno. 

![img_12.png](ims%2FW4%2Fimg_12.png)

Pero desafortunadamente es demasiado adecuado cuando trazamos las predicciones contra el conjunto de validación, no vemos mucha mejora y de hecho nuestro MAE ha caído. Así que sigue siendo un paso en la dirección correcta y considere una arquitectura como esta a medida que avanza, pero tal vez necesite modificar algunos de los parámetros para evitar el overfittingng.

![img_13.png](ims%2FW4%2Fimg_13.png)

Algunos de los problemas son claramente visualizar cuando trazamos la pérdida contra el MAE, hay mucho ruido e inestabilidad allí. Una causa común para picos pequeños como ese es , un tamaño de lote pequeño que introduce más ruido aleatorio. 


![img_14.png](ims%2FW4%2Fimg_14.png)

No voy a entrar en los detalles aquí, pero si echa un vistazo a los videos de Andrew y su curso de optimización para el descenso de gradiente, hay algunas cosas realmente geniales allí. 

![img_15.png](ims%2FW4%2Fimg_15.png)

Una sugerencia era explorar el tamaño del lote y asegurarse de que sea apropiado para mis datos. Así que en este caso vale la pena experimentar con diferentes tamaños de lote. Así que, por ejemplo, experimentó con diferentes tamaños de lote tanto más grande y más pequeño que el original 32, y cuando probé 16 se puede ver el impacto aquí en el conjunto de validación, y aquí en la pérdida de entrenamiento y datos MAE.

![img_16.png](ims%2FW4%2Fimg_16.png)

Así que combinando CNN y LSTM hemos sido capaces de construir nuestro mejor modelo hasta ahora, a pesar de algunos bordes rugosos que podrían ser refinados.

![img_17.png](ims%2FW4%2Fimg_17.png)

En el siguiente video, pasaremos a través de un cuaderno que entrena con este modelo para que puedas verlo por ti mismo, y también quizás aprenderemos a hacer algunas afinación para mejorar aún más el modelo. Después de eso, tomaremos la arquitectura del modelo y la aplicaremos a algunos datos del mundo real en lugar de los sintéticos que has estado usando todo el tiempo.

## More on Batch Sizing
[<- Return to INDEX 4](#index-4)

[Aquí](https://www.youtube.com/watch?v=4qJaSmvhxi8)
 encontrará más información

## Convolutions with LSTM (Lab 1)
[<- Return to INDEX 4](#index-4)

El siguiente vídeo es un screencast que demuestra cómo utilizar las capas de convolución con la red LSTM que construyó la semana pasada. Puede encontrar el enlace al cuaderno 
aquí [C4_W4_Lab_1_LSTM.ipynb](notebooks%2FW4%2FC4_W4_Lab_1_LSTM.ipynb)
.

## Convolutions with LSTM
[<- Return to INDEX 4](#index-4)

![img_18.png](ims%2FW4%2Fimg_18.png)

En el video anterior, viste cómo puedes apilar una capa convolucional con LSTM y LSTM bidireccionales para hacer predicción de secuencias. En este video, vamos a revisar un libro de trabajo para esto, que luego puedes probar por ti mismo más tarde. Como siempre, vamos a comprobar si tenemos tensorflow instalado. Si no es así, instale la última versión cada noche o la última versión de tensorflow.org.

![img_19.png](ims%2FW4%2Fimg_19.png)

Una vez que sepas que tienes la versión dos, entonces este código generará la serie de tiempo sintética para ti, y este código convertirá la matriz de datos en un conjunto de datos para entrenamiento. Tenga en cuenta que hemos ampliado las dimensiones en la primera línea. Esta función auxiliar puede realizar la previsión para nosotros después del entrenamiento.

Vamos a entrenar para 500 Épocas ahora y mantener un ojo en nuestra pérdida y nuestra mae. Cuando esté hecho, ejecutaremos nuestro pronóstico y trazaremos los resultados. Ya la curva se ve mucho mejor y la meseta que habíamos visto en entrenamiento previo ha desaparecido, y nuestra mae es baja, es justo por encima de cinco en el conjunto de validación. 

![img_20.png](ims%2FW4%2Fimg_20.png)

Finalmente, si trazamos nuestra pérdida de entrenamiento y mae, también vemos una curva saludable hacia abajo.

## Real Data - Sunspots
[<- Return to INDEX 4](#index-4)

![img_21.png](ims%2FW4%2Fimg_21.png)

En las últimas semanas, miró los datos de series temporales y examinó algunas técnicas para pronosticar esos datos incluyendo análisis estadístico, regresión lineal y aprendizaje automático con tanto redes de aprendizaje profundo como redes neuronales recurrentes.

Pero ahora vamos a ir más allá de los datos sintéticos a algunos datos del mundo real y aplicar lo que hemos aprendido a crear pronósticos para ello. 

![img_22.png](ims%2FW4%2Fimg_22.png)

Comencemos con este conjunto de datos de Kaggle, que rastrea las manchas solares mensualmente desde 1749 hasta 2018. Las manchas solares tienen ciclos estacionales aproximadamente cada 11 años. Así que probemos esto para ver si podemos predecir a partir de él. 

![img_23.png](ims%2FW4%2Fimg_23.png)

Es un conjunto de datos CSV con la primera columna siendo un índice, la segunda es una fecha en formato año, mes, día, y la tercera es la fecha de ese mes en que se tomó la medición. Es una cantidad mensual promedio que debería ser al final de ese mes. 

![img_24.png](ims%2FW4%2Fimg_24.png)

Puedes descargarlo desde Kaggle o si estás usando el cuaderno en esta lección, Lo he alojado convenientemente para tu en mi Cloud Storage. Es un conjunto de datos bastante simple, pero nos ayuda a entender un poco más sobre cómo optimizar nuestro código para predecir el conjunto de datos basado en la naturaleza de sus datos subyacentes. 

Por supuesto, un tamaño no se ajusta a todos , especialmente cuando se trata de datos que tienen estacionalidad. Así que echemos un vistazo al código. De acuerdo, antes que nada, si estás usando un codelab, entonces necesitarás obtener los datos en tu instancia de codelab. 

![img_25.png](ims%2FW4%2Fimg_25.png)

Este código descargará el archivo que he almacenado para usted. Realmente deberías obtenerlo de Kaggle y almacenarlo en tu propio servidor o incluso subirlo manualmente a codelab, pero para mayor comodidad, lo he almacenado aquí.

![img_26.png](ims%2FW4%2Fimg_26.png)

Aquí está el código para leer el archivo CSV y obtener sus datos en una lista de manchas solares y marcas de tiempo. Comenzaremos importando la biblioteca CSV. Entonces abriremos el archivo. Si está utilizando codelab y W obtiene el código que vio anteriormente, descarga el CSV y lo coloca en la temperatura de barra. Así que este código simplemente lo lee de ahí. 

Esta línea, siguiente lector, se llama Antes de recorrer las filas y el lector, y simplemente lee la primera línea y terminamos tirándola. Esto se debe a que los títulos de columna están en la primera línea del archivo como se puede ver aquí. Luego, veremos a través de el lector leyendo el archivo línea por línea. Nuestras manchas solares están en realidad en la columna 2 y queremos que se conviertan en un float. 

A medida que se lee el archivo, cada elemento se leerá como una cadena, así que también podemos convertirlos ahora en lugar de iterar a través de la lista más adelante y luego convertir todos los tipos de datos. Del mismo modo, leeremos los pasos de tiempo como enteros. 

![img_27.png](ims%2FW4%2Fimg_27.png)

Como gran parte del código que vamos a utilizar para procesar estos acuerdos con matrices NumPy, también podemos ahora convertir una lista a matrices NumPy. Es más eficiente hacerlo de esta manera, acumule sus datos en una lista desechable y luego convertirlo a NumPy de lo que habría sido para comenzar con matrices NumPy, porque cada vez que agrega un elemento a un NumPy, hay una gran cantidad de administración de memoria pasando a clonar la lista, tal vez una gran cantidad de datos que pueden ser lentos. 

![img_28.png](ims%2FW4%2Fimg_28.png)

Si trazamos nuestros datos se ve así. Tenga en cuenta que tenemos estacionalidad, pero no es muy regular con algunos picos y mucho más altos que otros. También tenemos un poco de ruido, pero no hay tendencia general. 

![img_29.png](ims%2FW4%2Fimg_29.png)

Como antes, dividamos nuestra serie en un conjunto de datos de capacitación y validación. Nos dividiremos a la hora 1.000.

![img_30.png](ims%2FW4%2Fimg_30.png)

Tendremos un tamaño de ventana de 20, un tamaño de lote de 32 y un búfer barajado de 1.000. Usaremos el mismo código de conjunto de datos de ventana que hemos estado usando toda la semana para convertir una serie en un conjunto de datos en el que podemos entrenar.

## Train and Tune the Model
[<- Return to INDEX 4](#index-4)

![img_31.png](ims%2FW4%2Fimg_31.png)

Volveremos a la simple DNN que vimos en la segunda semana para el entrenamiento y veremos qué pasa. 

![img_32.png](ims%2FW4%2Fimg_32.png)

Obtenemos un gráfico como este, que al menos para el globo ocular se ve muy bien, pero tiene un MAE muy grande así que algo debe estar mal. 

![img_33.png](ims%2FW4%2Fimg_33.png)

De hecho, si nos acercamos a los resultados podemos ver con un poco más de detalle acerca de cómo se comporta el pronóstico en los datos originales. Nuestra pista del problema podría ser nuestro tamaño de ventana.

![img_34.png](ims%2FW4%2Fimg_34.png)

Recuerde que antes dijimos que es un 20, así que nuestros tamaños de ventana de entrenamiento son 20 rebanadas de tiempo de datos. Y dado que cada rebanada de tiempo es un mes en tiempo real nuestra ventana es un poco menos de dos años.

![img_35.png](ims%2FW4%2Fimg_35.png)

Pero si recuerdan este gráfico, podemos ver que la estacionalidad de las manchas solares es mucho mayor que dos años. Está más cerca de 11 años. Y en realidad alguna ciencia nos dice que incluso podrían ser 22 años con diferentes ciclos interligados entre sí. 

![img_36.png](ims%2FW4%2Fimg_36.png)

Entonces, ¿qué pasaría si nos reentrenamos con un tamaño de ventana de 132, que es 11 años de datos como nuestro tamaño de ventana?

![img_37.png](ims%2FW4%2Fimg_37.png)

Ahora bien, aunque este gráfico tiene un aspecto similar, podemos ver en el MAE que en realidad empeoró , por lo que aumentar el tamaño de la ventana no funcionó. ¿Por qué crees que sería?

![img_38.png](ims%2FW4%2Fimg_38.png)

Bueno, mirando hacia atrás a los datos, podemos darnos cuenta de que es estacional a alrededor de 11 años, pero no necesitamos una temporada completa en nuestra ventana. Acercando los datos de nuevo, veremos algo como esto donde es solo la serie temporal típica.

Los valores más adelante están algo relacionados con los anteriores, pero eso es mucho ruido. Así que tal vez no necesitamos una gran ventana de tiempo para entrenar. Tal vez deberíamos ir con algo un poco más como nuestro inicial 20, vamos a probar 30.

Así que si miramos hacia atrás en este código, podemos cambiar nuestro tamaño de ventana a 30. Pero luego mira el tiempo dividido, el conjunto de datos tiene alrededor de 3.500 elementos de datos, pero lo estamos dividiendo en capacitación y validación. Ahora 1.000, lo que significa sólo 1.000 para la capacitación y 2.500 para la validación. 

![img_39.png](ims%2FW4%2Fimg_39.png)

Esa es una división muy mala. No hay suficientes datos de entrenamiento. Así que hagámoslo 3.500 en su lugar. Y luego cuando nos entrenemos, obtendremos esto. 

![img_40.png](ims%2FW4%2Fimg_40.png)

Nuestro MAE ha mejorado a 15 pero ¿podemos hacerlo aún mejor? Bueno, una cosa que podemos intentar es editar el diseño de la red neuronal y altura de parámetros. 

![img_41.png](ims%2FW4%2Fimg_41.png)

Si recuerdas, teníamos tres capas de 10, 10 y 1 neuronas. Nuestra forma de entrada es ahora más grande a los 30. 

![img_42.png](ims%2FW4%2Fimg_42.png)

Así que tal vez intente diferentes valores aquí, como 30, 15 y 1, y vuelva a entrenar.

![img_43.png](ims%2FW4%2Fimg_43.png)
Sorprendentemente, esto fue un pequeño paso hacia atrás, con nuestro MAE aumentando.

![img_45.png](ims%2FW4%2Fimg_45.png)

Tampoco valía la pena el tiempo de cálculo adicional para las capas de neuronas adicionales. Así que volvamos a 10, 10, 1 y en su lugar veamos la tasa de aprendizaje.

Vamos a ajustarlo un poco.

![img_44.png](ims%2FW4%2Fimg_44.png)

Ahora después del reentrenamiento, puedo ver que mi MAE ha disminuido un poco, lo cual es bueno.

## Prediction
[<- Return to INDEX 4](#index-4)

![img_46.png](ims%2FW4%2Fimg_46.png)

Por interés, hagamos una predicción. El tamaño de la ventana que estoy usando es de 30 pasos, y el conjunto de datos tiene 3.235 pasos de largo. Entonces, si quiero predecir el siguiente valor después del final de mi conjunto de datos, usaría este código.

![img_48.png](ims%2FW4%2Fimg_48.png)

. Y obtendría el resultado 7.0773993.

![img_49.png](ims%2FW4%2Fimg_49.png)

El conjunto de datos va hasta julio de 2018, así que estoy prediciendo 7.077 manchas solares para agosto de 2018. Y si miro este gráfico de observaciones, que tiene algunos datos ligeramente diferentes de mi conjunto de datos, puedo ver que el número real registrado de manchas solares en agosto de 2018 fue de 8.7. Así que la predicción no es tan mala, pero veamos si podemos mejorarla.

![img_47.png](ims%2FW4%2Fimg_47.png)

Con estas configuraciones, bajé el MAE a 13.75, y la predicción fue 8.13, que está mucho más cerca de la lectura real real real de 8.7. Sin embargo, hay elementos aleatorios en la creación de modelos, por lo que sus resultados pueden variar. Hacer precisión basada en una sola predicción como esta también es una receta para decepción, y estás mucho mejor evaluando la precisión media sobre una serie de lecturas.

![img_50.png](ims%2FW4%2Fimg_50.png)

Así que aquí, analizamos el uso de un DNN para predecir los valores de las manchas solares. Con un poco de afinación, hemos reducido un poco nuestro MAE. Y cuando intentamos predecir el valor del próximo mes usando este modelo, nos acercamos bastante al valor real. En el siguiente vídeo, recorrerás este libro de trabajo para verlo en acción. Y entonces deberías probarlo por ti mismo, para ver si puedes mejorar en lo que tengo. Después de eso, mirarás en los RNN otra vez y verás si puedes obtener mejores predicciones usando esos.

## Sunspots Notebooks (Lab 2 and Lab 3)
[<- Return to INDEX 4](#index-4)

Los próximos screencasts mostrarán cómo puede utilizar diferentes arquitecturas de red para predecir datos del mundo real.

Aquí [C4_W4_Lab_2_Sunspots_DNN.ipynb](notebooks%2FW4%2FC4_W4_Lab_2_Sunspots_DNN.ipynb)
 está el cuaderno sólo para DNN.

Aquí [C4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb](notebooks%2FW4%2FC4_W4_Lab_3_Sunspots_CNN_RNN_DNN.ipynb)
 está la versión con las capas combinadas.

## Sunspots
[<- Return to INDEX 4](#index-4)

![img_51.png](ims%2FW4%2Fimg_51.png)

En la lección anterior, observaste el uso de un conjunto de datos del mundo real, uno con lecturas de la actividad de las manchas solares durante un par de cientos de años. Y viste cómo construir un DNN simple para predecir la actividad usando ese conjunto de datos. Ahora echemos un vistazo al cuaderno y lo veremos en acción por nosotros mismos.

![img_52.png](ims%2FW4%2Fimg_52.png)

Como siempre, vamos a comprobar si tenemos instalado TensorFlow 2.
Y lo hacemos solo podemos progresar. Este código solo importa numpy y pyplot y crea una función auxiliar para gráficos

![img_53.png](ims%2FW4%2Fimg_53.png)

Este código descarga el conjunto de datos y lo coloca en el directorio temporal. Es un conjunto de datos muy pequeño, por lo que debería ejecutarse muy rápidamente.

![img_54.png](ims%2FW4%2Fimg_54.png)

Ahora importaremos los datos del CSV y crearemos matrices numpy desde él. Podemos trazar los datos para verlo estacionalidad y su ruido, pero no hay tendencia notable.

![img_55.png](ims%2FW4%2Fimg_55.png)

Ahora dividiremos los datos y configuraremos nuestras constantes. Lo más notable es el tamaño de la ventana que he establecido en 60. 


![img_56.png](ims%2FW4%2Fimg_56.png)

Este código es una función auxiliar para convertir los datos en un conjunto de datos de ventana.


![img_57.png](ims%2FW4%2Fimg_57.png)

Y ahora vamos a entrenar con un DNN de tres capas que tiene 2010, y una neuronas.

![img_58.png](ims%2FW4%2Fimg_58.png)

Utilizaremos el descenso de gradiente estocástico como nuestro optimizador con una tasa de aprendizaje de 1e-7. Una vez finalizado el entrenamiento, podemos obtener pronósticos en todo el conjunto de validación. Y nuestra trama se ve bastante bien.


Si mencionamos MAE, estamos a unos 15.

![img_59.png](ims%2FW4%2Fimg_59.png)


Debe obtener valores ligeramente diferentes debido a la aleatoriedad de la inicialización del modelo y el descenso del gradiente estocástico. Así que no te preocupes si está un poco apagado. Si está muy apagado, vuelva y revise su código o reinicie el tiempo de ejecución e inténtelo de nuevo.

## Combining our Tools for Analysis
[<- Return to INDEX 4](#index-4)

![img_65.png](ims%2FW4%2Fimg_65.png)

Así que hemos mirado los datos de manchas solares usando un DNN estándar como al principio de este curso. También hemos estado trabajando mucho con RNN y un poco con redes neuronales convolucionales. Entonces, ¿qué pasaría si los juntamos a todos para ver si podemos predecir la actividad de la mancha solar?


Este es un conjunto de datos difícil porque como ya hemos visto, aunque es estacional el período es realmente largo, alrededor de 11 años, y no es perfectamente sazonable durante ese período. Así que echemos un vistazo al uso de todas las herramientas que tenemos para ver si podemos construir una predicción decente usando el aprendizaje automático. 

![img_66.png](ims%2FW4%2Fimg_66.png)

Así que aquí está la primera pieza de código que podemos probar. Me he vuelto un poco loco aquí, así que vamos a descomponerlo pieza por pieza. En primer lugar, estoy configurando el tamaño del lote en 64 y el tamaño de la ventana en 60. 

Entonces comenzaremos con una convolución 1D que aprenderemos 32 filtros. Esto dará salida a un par de LSTM con 32 células cada una antes de alimentar a un DNN similar a lo que vimos anteriormente, 30 neuronas, luego 10, y una. 

Finalmente, como nuestros números están en el rango 1-400, hay una capa Lambda que multiplica nuestra X por 400. 

![img_67.png](ims%2FW4%2Fimg_67.png)


Con la primera prueba para establecer el mejor aprendizaje, la tasa obtenemos este gráfico. Esto sugiere que la mejor tasa de aprendizaje para esta red será de alrededor de 10 a menos 5. 

![img_68.png](ims%2FW4%2Fimg_68.png)

 Así que cuando entrené para 500 épocas con esta configuración, aquí están mis resultados. 

![img_69.png](ims%2FW4%2Fimg_69.png)

Es bastante bueno con un buen MAE bajo.

![img_70.png](ims%2FW4%2Fimg_70.png)

Pero cuando miro mi función de pérdida durante el entrenamiento, puedo ver que hay mucho ruido que me dice que sin duda puedo optimizarlo un poco, y como vimos en videos anteriores, una de las mejores cosas a mirar en estas circunstancias es el tamaño del lote.

![img_71.png](ims%2FW4%2Fimg_71.png)

Así que lo aumentaré a 256 y volver a entrenar. 

![img_72.png](ims%2FW4%2Fimg_72.png)

Después de 500 épocas, mis predicciones han mejorado un poco lo cual es un paso en la dirección correcta. 

![img_73.png](ims%2FW4%2Fimg_73.png)

Pero mira mi ruido de entrenamiento. Particularmente hacia el final de la formación es realmente ruidoso, pero es una onda de aspecto muy regular. Esto sugiere que mi tamaño de lote más grande era bueno, pero tal vez un poco apagado. 

![img_74.png](ims%2FW4%2Fimg_74.png)

¿Qué pasaría si cambiara mis parámetros para que se adaptara, y no solo la ventana y el tamaño del lote, qué tal si cambiara los filtros también? Entonces, ¿qué pasa si configuro eso en 60, y los LSTM en 60 en lugar de 32 o 64? Mi DNN ya se ve bien, así que no los cambiaré.

![img_75.png](ims%2FW4%2Fimg_75.png)

Experimentar con hiperparámetros como este es una gran manera de aprender los ins y outs del aprendizaje automático, no solo con secuencias sino con cualquier cosa. Recomiendo a fondo pasar tiempo en y ver si puedes mejorar en este modelo. 

Además, usted debe acompañar que trabajan con profundizar en cómo todas estas cosas en el aprendizaje automático trabajan y cursos de Andrews son geniales para eso. Les recomiendo encarecidamente si aún no los has hecho.

## Week 4 Quiz
[<- Return to INDEX 4](#index-4)

1. **Question 1**  
How do you add a 1 dimensional convolution to your model for predicting time series data?

   - [ ] Use a 1DConvolution layer type
   - [x] Use a Conv1D layer type
   - [ ] Use a 1DConv layer type
   - [ ] Use a ConvolutionD1 layer type

   > **Correct!** The correct syntax for adding a one-dimensional convolution layer in many deep learning frameworks is `Conv1D`.

2. **Question 2**  
What’s the input shape for a univariate time series to a Conv1D?

   - [ ] [1]
   - [ ] []
   - [x] [None, 1]
   - [ ] [1, None]

   > **Correct!** The standard input shape for a univariate time series in `Conv1D` is `[None, 1]`, allowing any length of sequence with one feature.

3. **Question 3**  
You used a sunspots dataset that was stored in CSV. What’s the name of the Python library used to read CSVs?

   - [ ] PyFiles
   - [ ] PyCSV
   - [ ] CommaSeparatedValues
   - [x] CSV

   > **Correct!** Python's CSV library is commonly used for reading and writing CSV files. It provides functionality for handling CSV formatted data.

4. **Question 4**  
If your CSV file has a header that you don’t want to read into your dataset, what do you execute before iterating through the file using a ‘reader’ object?

   - [ ] reader.next
   - [x] next(reader)
   - [ ] reader.read(next)
   - [ ] reader.ignore_header()

   > **Correct!** The function `next(reader)` skips the current line where the reader is positioned, commonly used to skip headers in CSV files.

5. **Question 5**  
When you read a row from a reader and want to cast column 2 to another data type, for example, a float, what’s the correct syntax?

   - [ ] You can’t. It needs to be read into a buffer and a new float instantiated from the buffer
   - [ ] Convert.toFloat(row[2])
   - [x] float(row[2])
   - [ ] float f = row[2].read()

   > **Correct!** To cast a value to a float in Python, you straightforwardly use `float(row[2])`.

6. **Question 6**  
What was the sunspot seasonality?

   - [ ] 4 times a year
   - [x] 11 or 22 years depending on who you ask
   - [ ] 11 years
   - [ ] 22 years

   > **Correct!** Sunspot activity varies, and the seasonality can be described as occurring approximately every 11 or 22 years depending on different studies.

7. **Question 7**  
After studying this course, what neural network type do you think is best for predicting time series like our sunspots dataset?

   - [x] A combination of all other answers
   - [ ] Convolutions
   - [ ] RNN / LSTM
   - [ ] DNN

   > **Correct!** Often, a combination of several neural network types can yield the best results for complex time series predictions like those involving sunspots.

8. **Question 8**  
Why is MAE a good analytic for measuring accuracy of predictions for time series?

   - [ ] It punishes larger errors
   - [ ] It biases towards small errors
   - [x] It doesn’t heavily punish larger errors like square errors do
   - [ ] It only counts positive errors

   > **Correct!** Mean Absolute Error (MAE) is beneficial because it gives a straightforward average of absolute errors and does not overly penalize larger errors compared to metrics like the Mean Squared Error.

## Lecture Notes Week 4
[<- Return to INDEX 4](#index-4)

Los apuntes de las conferencias están disponibles en nuestra plataforma comunitaria. Si ya es miembro, inicie sesión en su cuenta y acceda a los apuntes de las conferencias 
aquí [C4_W4.pdf](notes%2FC4_W4.pdf)
.

## Reminder About the End of Access to Lab Notebooks
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

## Wrap Up
[<- Return to INDEX 4](#index-4)

¡Enhorabuena por haber llegado hasta aquí! En este curso, ¡se ha sumergido en el agua de la comprensión de las secuencias y de algunas de las herramientas para predecirlas! Esperamos que haya sido un viaje divertido y ¡estamos deseando ver lo que construye a continuación! :)  

## Congratulations
[<- Return to INDEX 4](#index-4)

Así que eso te lleva al final de este curso sobre secuencias y predicción. Espero que hayas disfrutado viendo la miríada de diferentes maneras en que puedes usar la máquina aprendiendo para hacer predicciones basadas en datos secuenciales y de series temporales. Dicho esto, realmente sólo has arañado la superficie de lo que es posible, y sólo has dado los primeros pasos en tu viaje. 

![img_60.png](ims%2FW4%2Fimg_60.png)

Hay mucho más que aprender, pero con lo que has logrado en las últimas semanas, estás bien equipado para profundizar y mirar escenarios más complejos, incluyendo cosas como datos multivariados de series temporales. Realmente esperamos que hayas disfrutado de este curso, y como siempre nos encantaría escuchar tus comentarios a medida que creamos más. - Gracias. - Gracias.

## References
[<- Return to INDEX 4](#index-4)

Se trata de una recopilación de los recursos que aparecen en los vídeos de las clases, los laboratorios no calificados y las tareas.

**Semana 1:**

- [Ley
 de Moore ](https://en.wikipedia.org/wiki/Moore%27s_law)(Wikipedia)

- [Correlaciones espurias: Arcadas vs Doctorados](https://tylervigen.com/view_correlation?id=97)
 (Tyler Vigen)

- [Tasa de natalidad y mortalidad en Japón](https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/Bdrates_of_Japan_since_1950.svg/450px-Bdrates_of_Japan_since_1950.svg.png)
 (Wikipedia)

- [Temperatura global y dióxido de carbono](https://www.globalchange.gov/browse/multimedia/global-temperature-and-carbon-dioxide)
 (GlobalChange.gov)

- [Forma pendiente-intercepto](https://en.wikipedia.org/wiki/Linear_equation#Slope%E2%80%93intercept_form_or_Gradient-intercept_form)
 (Wikipedia)

- [API de Tensorflow](https://www.tensorflow.org/api_docs/python/tf)
 (Documentación de TF)

- [Numpy](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.html)
 (Sitio Web oficial)

- [Pyplot](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.html)
 (Sitio Web oficial)

- [Métricas Keras](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/)
 (Sitio Web oficial)

**Semana 2:**

- [aPI
tf
.data ](https://www.tensorflow.org/guide/data)(Documentación TF)

- [tf.data
.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
 (Documentación TF)

- [Aplanar un conjunto de datos de ventanas](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flatten_a_dataset_of_windows_2)
 (Documentación TF)

- [LearningRateScheduler](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler)
 (Documentación TF)

**Semana 3:**

- [Pérdida de Huber](https://en.wikipedia.org/wiki/Huber_loss)
 (Wikipedia)

- [SimpleRNN](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)
 (Documentación TF)

- [Capa lambda](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda)
 (Documentación TF)

- [Funciones de activación](https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions)
 (Wikipedia)

- [LSTM](https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay)
 (DeepLearning.AI)

- [Capa
LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
 (Documentación TF)

**Semana 4:**

- [Redes neuronales convolucionales](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome)
 (DeepLearning.AI)

- [Mini-lote de descenso gradiente](https://www.youtube.com/watch?v=4qJaSmvhxi8&ab_channel=DeepLearningAI)
 (DeepLearning.AI)

- [Conjunto de datos de manchas solares](https://www.kaggle.com/datasets/robervalt/sunspots)
 (Robert Valt)

- [Condiciones solares](https://sws.bom.gov.au/Solar/1/6)
 (Oficina de Meteorología de Australia)

- [Temperaturas mínimas diarias en Melbourne](https://github.com/jbrownlee/Datasets/blob/master/daily-min-temperatures.csv)
 (presentado por Jason Brownlee, fuente: Oficina Australiana de Meteorología)

## Acknowledgments
[<- Return to INDEX 4](#index-4)

Además de los desarrolladores originales del plan de estudios, las siguientes personas hicieron contribuciones 
significativas a la actualización del Curso 4:

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


## Specialization wrap up A conversation with Andrew Ng
[<- Return to INDEX 4](#index-4)

![img_61.png](ims%2FW4%2Fimg_61.png)

Felicitaciones por llegar al final de este TensorFlow y la práctica de especialización. En los cuatro cursos, aprendes cómo construir una red neuronal profunda, cómo construir una red comf, cómo construir modelos de secuencia para PNL y finalmente, cómo construir modelos de series temporales. Has construido y has aprendido mucho de cosas en la especialización, y espero que te des una palmadita en la espalda para llegar al final. 

ealmente hemos recorrido un largo camino, si recuerdas nuestra primera lección es que hicimos dos conjuntos de números donde la relación era y es igual a 2x menos 1. Tuvimos un poco de DNN que la gente usa entonces para aprender eso y para hacer ese patrón de coincidencia de la x contra y, pero desde entonces, hemos ido a la visión por ordenador, procesamiento de lenguaje natural, modelado de secuencia de tiempo, manchas solares, todo eso. 

![img_62.png](ims%2FW4%2Fimg_62.png)

Así que realmente creo que realmente has sólo dado el primer paso, aunque en un viaje mucho más grande, hay muchas más cosas que aprender. Una de las piezas de retroalimentación que recibimos de estudiantes es que realmente les encanta esto, pero quieren aprender más. 

Yo siempre recomiendo que las otras especializaciones que Andrew ha hecho de las que he personalmente aprendí también. Así que si realmente quieres entrar en más profundidad en, por ejemplo, cómo funciona un LSTM o cómo funciona un RNN, o cómo incluso funciona una convolución, hay un gran material por ahí, y hemos diseñado estos materiales para malla entre sí. 

![img_63.png](ims%2FW4%2Fimg_63.png)

Así que si realmente quieres aprender en profundidad, Andrew es el maestro, y si quieres aprender a aplicar eso al código y cómo tomar eso, y ponerlo en práctica, entonces este curso y futuros cursos en los que estamos trabajando serán realmente alrededor de eso. 

Así que esperamos verte allí. Como dijo Lawrence, espero que esto sea solo una parte de su viaje para llegar a ser grande en el aprendizaje profundo. La especialización de aprendizaje profundo complementa esta, y así que si quieres profundizar en mayor profundidad, puedes comprobarlo también. Todavía hay más por aprender. Así que Lawrence y todavía voy a seguir a construir especializaciones adicionales para ayudar a aprender aún más sobre TensorFlow. 

![img_64.png](ims%2FW4%2Fimg_64.png)

Mirando hacia adelante. Trabajar con Lawrence ha sido el punto culminante de estos últimos meses y espero adelante construir más cursos juntos. A todos ustedes que se quedaron con nosotros a lo largo de todo este período, gracias por todo el trabajo duro que han puesto en aprender estas cosas y seguir viendo el espacio, vamos a tener más cosas para ustedes en el futuro. Gracias.

## What next?
[<- Return to INDEX 4](#index-4)

¿Listo para desplegar sus modelos por el mundo? Aprenda a poner en marcha sus modelos con la 
[TensorFlow: Especialización en datos y despliegue](https://bit.ly/3ojuT1o)
.

¿Busca personalizar y construir potentes modelos del mundo real para escenarios complejos? Eche un vistazo a la 
[TensorFlow: Especialización en Técnicas Avanzadas](https://bit.ly/39iAsZQ)
. 

## Opportunity to Mentor Other Learners
[<- Return to INDEX 4](#index-4)

¡Nuestro más sincero agradecimiento por inscribirse en esta especialización y formar parte de nuestra comunidad global de estudiantes!


Si le apasiona ser voluntario y devolver algo a la comunidad guiando a otros alumnos para que completen esta especialización y/o prueben otras especializaciones publicadas por 
[Deeplearning](https://forms.gle/YHBfNquA8W4NA6TU8)
.IA en Coursera, inscríbase 
[en el Programa de la Comunidad DeepLearning.I
A.](https://forms.gle/YHBfNquA8W4NA6TU8)


Alguien del personal de DeepLearning.IA se pondrá en contacto con usted cuando haya puestos disponibles adecuados a su perfil.


Nota: No es obligatorio inscribirse. Se trata de un servicio puramente opcional y voluntario.

# Weekly Assignment Adding Convolutions
[<- Return to INDEX 0](#index-0)

## Adding CNNs to improve forecasts
[<- Return to INDEX 4](#index-4)
