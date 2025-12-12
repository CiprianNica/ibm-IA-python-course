SVM - Máquinas de vectores de soporte
Objetivo

El objetivo es implementar una función que:

Entrene un modelo de Máquina de Soporte Vectorial (SVM) usando SVC de sklearn.svm.

Realice predicciones en un conjunto de prueba.

Evalúe el modelo con las siguientes métricas:

Precisión (accuracy_score).

Matriz de confusión (confusion_matrix).

Reporte de clasificación (classification_report).

Devuelva los resultados en un diccionario.

Supervise la implementación con pruebas unitarias (unittest).



Instrucciones

Implementa una función llamada entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test) que:

Entrene un modelo SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42).

Prediga los valores de X_test.

Calcule las métricas de evaluación mencionadas.

Devuelva un diccionario con:

"predicciones": Array de predicciones del modelo.

"accuracy": Precisión del modelo en los datos de prueba.

"matriz_confusion": Matriz de confusión.

"reporte": Reporte de clasificación.

Usa el dataset de digits de sklearn.datasets, que contiene imágenes de números escritos a mano.

Asegúrate de que el modelo tenga al menos 90% de precisión en los datos de prueba.



Ejemplo de Uso

Una vez que la función esté implementada, el siguiente código debería ejecutarse correctamente:

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
 
# Cargar el dataset de dígitos escritos a mano
digits = load_digits()
X = digits.data  # Características (matriz de píxeles)
y = digits.target  # Etiquetas (números del 0 al 9)
 
# Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Llamar a la función y obtener las métricas
resultados = entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test)
 
# Mostrar los resultados
print("Precisión del modelo:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])


Salida esperada (aproximada)

Precisión del modelo: 0.9861111111111112
Matriz de Confusión:
 [[33  0  0  0  0  0  0  0  0  0]
 [ 0 28  0  0  0  0  0  0  0  0]
 [ 0  0 33  0  0  0  0  0  0  0]
 [ 0  0  0 33  0  1  0  0  0  0]
 [ 0  0  0  0 46  0  0  0  0  0]
 [ 0  0  0  0  0 46  1  0  0  0]
 [ 0  0  0  0  0  0 35  0  0  0]
 [ 0  0  0  0  0  0  0 33  0  1]
 [ 0  0  0  0  0  1  0  0 29  0]
 [ 0  0  0  0  0  0  0  1  0 39]]
Reporte de Clasificación:
               precision    recall  f1-score   support
 
           0       1.00      1.00      1.00        33
           1       1.00      1.00      1.00        28
           2       1.00      1.00      1.00        33
           3       1.00      0.97      0.99        34
           4       1.00      1.00      1.00        46
           5       0.96      0.98      0.97        47
           6       0.97      1.00      0.99        35
           7       0.97      0.97      0.97        34
           8       1.00      0.97      0.98        30
           9       0.97      0.97      0.97        40
 
    accuracy                           0.99       360
   macro avg       0.99      0.99      0.99       360
weighted avg       0.99      0.99      0.99       360