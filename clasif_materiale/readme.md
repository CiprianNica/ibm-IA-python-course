Clasificador inteligente de materiales reciclables
Contexto:
Imagina que trabajas para una empresa de reciclaje inteligente.

Tu tarea consiste en diseñar un sistema que pueda predecir automáticamente si un objeto es papel, plástico o metal, a partir de sus propiedades físicas, usando el algoritmo de k vecinos más cercanos (KNN).

Vas a utilizar Python con las librerías numpy, pandas, matplotlib y sklearn para entrenar y visualizar el modelo.



📦 Objetivo

Implementa las siguientes clases:

1. RecyclableItem

Representa un objeto reciclable con tres atributos:

weight: peso del objeto en gramos.

volume: volumen en cm³.

material_type: tipo de material codificado como:

0 para papel

1 para plástico

2 para metal

Método necesario:

to_vector(self): devuelve [weight, volume], útil para alimentar el modelo.



2. RecyclingDataGenerator

Genera objetos sintéticos para entrenar el modelo.

Métodos:

__init__(self, num_samples=150): constructor de la clase:

num_samples: número total de objetos a generar (repartidos entre los tres tipos de material).

generate(self): genera y devuelve una lista de objetos RecyclableItem con las siguientes características:

Papel (0):

Peso: media ≈ 30 g → np.random.normal(30, 5)

Volumen: media ≈ 250 cm³ → np.random.normal(250, 30)

Plástico (1):

Peso: media ≈ 80 g → np.random.normal(80, 10)

Volumen: media ≈ 150 cm³ → np.random.normal(150, 20)

Metal (2):

Peso: media ≈ 150 g → np.random.normal(150, 20)

Volumen: media ≈ 80 cm³ → np.random.normal(80, 10)

3. RecyclableMaterialClassifier

Clasificador que entrena un modelo de KNN.

Métodos:

Constructor de la clase  __init__(self, k=5):

k: número de vecinos más cercanos a usar (por defecto: 5)

fit(records): entrena el modelo con una lista de objetos RecyclableItem.

predict(weight, volume): devuelve el tipo de material predicho (0, 1 o 2) para un nuevo objeto.

evaluate(records): imprime métricas de clasificación (classification_report, confusion_matrix) con un conjunto de prueba.



4. RecyclablePredictionExample

Clase que coordina todo el flujo:

Genera los datos.

Separa en entrenamiento y prueba.

Entrena el clasificador.

Evalúa el rendimiento.

Hace una predicción para un nuevo objeto (por ejemplo, peso = 60, volumen = 180).

Visualiza los datos y las predicciones en un gráfico 2D con colores distintos para cada tipo de material.



✅ Ejemplo de uso

example = RecyclablePredictionExample()
example.run()


Salida esperada

[[10  0  0]
 [ 0 18  0]
 [ 0  0 17]]
              precision    recall  f1-score   support
 
           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00        18
           2       1.00      1.00      1.00        17
 
    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
 
 
📦 Predicción para un nuevo objeto:
   Peso: 60g, Volumen: 180cm³
   Tipo estimado: Plástico
