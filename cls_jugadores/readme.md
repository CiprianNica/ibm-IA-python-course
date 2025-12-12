Clasificador de jugadores de baloncesto
Objetivo del ejercicio:

Tu misión es construir un modelo inteligente que clasifique a jugadores de baloncesto según su rendimiento en tres categorías: "Bajo", "Medio" y "Alto", utilizando para ello sus características físicas y estadísticas de juego.

Usarás el algoritmo de árboles de decisión junto con NumPy, pandas, matplotlib y scikit-learn.



🎯 Contexto del problema

Un equipo de baloncesto ficticio está evaluando a nuevos jugadores y necesita una herramienta que, a partir de la altura, el peso y el promedio de puntos por partido, determine automáticamente el nivel de rendimiento del jugador.

Esta herramienta será clave para seleccionar a los mejores candidatos.



🧱 Estructura sugerida de la solución

1. BasketballPlayer

Una clase que representa a cada jugador. Sus atributos son:

height (int): altura en centímetros.

weight (int): peso en kilogramos.

avg_points (float): promedio de puntos por partido.

performance (str): nivel de rendimiento, con valores "Bajo", "Medio" o "Alto".

Método útil:

to_vector(): devuelve [height, weight, avg_points] para ser usado por el modelo.



2. BasketballDataGenerator

Una clase que genera datos sintéticos simulando jugadores reales.

Constructor

__init__(self, num_samples=200)

num_samples: número total de jugadores a generar (por defecto: 200).

Método clave:

generate(): devuelve una lista de objetos BasketballPlayer.

Cada jugador se genera con las siguientes características:

Altura (height): generada con una distribución normal de media 190 cm y desviación estándar 10 cm:

heights = np.random.normal(190, 10, self.num_samples)

Peso (weight): generada con una distribución normal de media 85 kg y desviación estándar 10 kg:

Promedio de puntos por partido (avg_points): generada con una distribución normal de media 10 puntos y desviación estándar 5:

El rendimiento del jugador se asigna de acuerdo al valor del promedio de puntos:

Menos de 8 puntos → "Bajo"

Entre 8 y 15 puntos → "Medio"

Más de 15 puntos → "Alto"



3. BasketballPerformanceClassifier

Encapsula el modelo de árbol de decisión. Métodos clave:

fit(players): entrena el modelo con una lista de jugadores.

predict(height, weight, avg_points): predice el rendimiento de un nuevo jugador.

evaluate(players): imprime la matriz de confusión y el informe de clasificación sobre un conjunto de prueba.



4. BasketballPredictionExample

Contiene el flujo principal:

Generar datos.

Dividirlos en entrenamiento y prueba.

Entrenar y evaluar el clasificador.

Hacer una predicción para un nuevo jugador (por ejemplo: altura = 198 cm, peso = 92 kg, puntos = 17).

Visualizar los jugadores usando matplotlib, diferenciando el rendimiento por colores.



📊 Visualización esperada

Un gráfico de dispersión donde cada punto representa un jugador.

El eje X muestra la altura y el eje Y el promedio de puntos.

El color indica el rendimiento:

🔴 Bajo

🟠 Medio

🟢 Alto

Además:

Se debe marcar la posición del jugador nuevo (por ejemplo, con una estrella ⭐ o un punto grande en azul).



✅ Requisitos técnicos

Usa NumPy para generar datos aleatorios.

Usa pandas para crear el DataFrame de visualización.

Usa DecisionTreeClassifier de sklearn.tree.

Representa visualmente los datos con matplotlib.



🧪 Ejemplo de uso

example = BasketballPredictionExample()
example.run()


Salida esperada

Confusion Matrix:
[[10  0  0]
 [ 0 23  0]
 [ 0  0 27]]
 
Classification Report:
              precision    recall  f1-score   support
 
        Alto       1.00      1.00      1.00        10
        Bajo       1.00      1.00      1.00        23
       Medio       1.00      1.00      1.00        27
 
    accuracy                           1.00        60
   macro avg       1.00      1.00      1.00        60
weighted avg       1.00      1.00      1.00        60
 
 
🎯 Predicción personalizada → Altura: 198 cm, Peso: 92 kg, Prom. puntos: 17
   → Categoría predicha: Alto
