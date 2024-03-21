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

## Preparing Features and Labels Notebook (Lab 1)
[<- Return to INDEX 2](#index-2)

## Preparing Features and Labels
[<- Return to INDEX 2](#index-2)

## Preparing Features and Labels (screencast)
[<- Return to INDEX 2](#index-2)

## Single Layer Network Notebook (Lab 2)
[<- Return to INDEX 2](#index-2)

## Feeding Windowed Dataset into Neural Network
[<- Return to INDEX 2](#index-2)

## Single Layer Neural Network
[<- Return to INDEX 2](#index-2)

## Machine Learning on Time Windows
[<- Return to INDEX 2](#index-2)

## Prediction
[<- Return to INDEX 2](#index-2)

## More on Single Layer Neural Network
[<- Return to INDEX 2](#index-2)

## Deep Neural Network Notebook (Lab 3)
[<- Return to INDEX 2](#index-2)

## Deep Neural Network Training, Tuning and Prediction
[<- Return to INDEX 2](#index-2)

## Deep Neural Network
[<- Return to INDEX 2](#index-2)

## Week 2 Quiz
[<- Return to INDEX 2](#index-2)

## Week 2 Wrap Up
[<- Return to INDEX 2](#index-2)

## Lecture Notes Week 2
[<- Return to INDEX 2](#index-2)

# Weekly Assignment Prediction with a DNN
[<- Return to INDEX 0](#index-0)

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

## Conceptual Overview
[<- Return to INDEX 3](#index-3)

## RNN Notebook (Lab 1)
[<- Return to INDEX 3](#index-3)

## Shape of the Inputs to the RNN
[<- Return to INDEX 3](#index-3)

## Outputting a Sequence
[<- Return to INDEX 3](#index-3)

## Lambda Layers
[<- Return to INDEX 3](#index-3)

## Adjusting the Learning Rate Dynamically
[<- Return to INDEX 3](#index-3)

## More Info on Huber Loss
[<- Return to INDEX 3](#index-3)

## LSTM
[<- Return to INDEX 3](#index-3)

## Link to the LSTM Lesson
[<- Return to INDEX 3](#index-3)

## LSTM Notebook (Lab 2)
[<- Return to INDEX 3](#index-3)

## Coding LSTMs
[<- Return to INDEX 3](#index-3)

## Week 3 Quiz
[<- Return to INDEX 3](#index-3)

## Week 3 Wrap Up
[<- Return to INDEX 3](#index-3)

## Lecture Notes Week 3
[<- Return to INDEX 3](#index-3)

# Weekly Assignment Using Layers for Sequence Processing
[<- Return to INDEX 0](#index-0)

## Forecast Using RNNs or LSTMs
[<- Return to INDEX 3](#index-3)

# Real-world Time Series Data
[<- Return to INDEX 0](#index-0)

## INDEX 4

- [A Conversation with Andrew Ng W4](#a-conversation-with-andrew-ng-w4)
- [Convolutions](#convolutions)
- [Convolutional Neural Network Course](#convolutional-neural-network-course)
- [Bi-directional LSTMs](#bi-directional-lstms)
- [More on Batch Sizing](#more-on-batch-sizing)
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

## Convolutions
[<- Return to INDEX 4](#index-4)

## Convolutional Neural Network Course
[<- Return to INDEX 4](#index-4)

## Bi-directional LSTMs
[<- Return to INDEX 4](#index-4)

## More on Batch Sizing
[<- Return to INDEX 4](#index-4)

## Convolutions with LSTM
[<- Return to INDEX 4](#index-4)

## Real Data - Sunspots
[<- Return to INDEX 4](#index-4)

## Train and Tune the Model
[<- Return to INDEX 4](#index-4)

## Prediction
[<- Return to INDEX 4](#index-4)

## Sunspots Notebooks (Lab 2 and Lab 3)
[<- Return to INDEX 4](#index-4)

## Sunspots
[<- Return to INDEX 4](#index-4)

## Combining our Tools for Analysis
[<- Return to INDEX 4](#index-4)

## Week 4 Quiz
[<- Return to INDEX 4](#index-4)

## Lecture Notes Week 4
[<- Return to INDEX 4](#index-4)

## Reminder About the End of Access to Lab Notebooks
[<- Return to INDEX 4](#index-4)

## Wrap Up
[<- Return to INDEX 4](#index-4)

## Congratulations
[<- Return to INDEX 4](#index-4)

## References
[<- Return to INDEX 4](#index-4)

## Acknowledgments
[<- Return to INDEX 4](#index-4)

## Specialization wrap up A conversation with Andrew Ng
[<- Return to INDEX 4](#index-4)

## What next?
[<- Return to INDEX 4](#index-4)

## Opportunity to Mentor Other Learners
[<- Return to INDEX 4](#index-4)

# Weekly Assignment Adding Convolutions
[<- Return to INDEX 0](#index-0)

## Adding CNNs to improve forecasts
[<- Return to INDEX 4](#index-4)
