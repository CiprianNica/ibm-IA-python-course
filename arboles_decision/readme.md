Arboles de decisión
🧠 Objetivo

Implementar una función en Python que entrene un modelo de árbol de decisión y lo utilice para predecir clases a partir de nuevos datos. Este ejercicio te ayudará a familiarizarte con los conceptos fundamentales de la clasificación supervisada usando scikit-learn.



🔧 Lo que debes hacer

Implementa una función llamada:

entrenar_arbol_decision(X_train, y_train, X_test)

La función debe cumplir con los siguientes pasos:

Crear un modelo de árbol de decisión usando DecisionTreeClassifier de sklearn.tree, con random_state=42 para garantizar resultados reproducibles.

Entrenar el modelo con los datos de entrada X_train (características) y y_train (etiquetas).

Hacer predicciones sobre los datos de prueba X_test.

Devolver un array de NumPy con las predicciones.



📥 Parámetros de entrada

X_train: array de NumPy con las características de entrenamiento (ejemplos ya conocidos).

y_train: array de NumPy con las clases o etiquetas correspondientes a X_train.

X_test: array de NumPy con nuevos ejemplos sin clasificar.



📤 Salida esperada

Un array de NumPy con las predicciones de clase para cada fila en X_test.



Ejemplo de Uso



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
 
# Cargar el dataset de flores Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Clases de las flores (Setosa, Versicolor, Virginica)
 
# Dividir en conjunto de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Llamar a la función que debes implementar
predicciones = entrenar_arbol_decision(X_train, y_train, X_test)
 
# Mostrar algunas predicciones
print("Predicciones del Árbol de Decisión:", predicciones[:10])
print("Valores reales:                    ", y_test[:10])


Salida esperada



Predicciones del Árbol de Decisión: [1 0 2 1 1 0 1 2 1 1]
Valores reales:                     [1 0 2 1 1 0 1 2 1 1]
