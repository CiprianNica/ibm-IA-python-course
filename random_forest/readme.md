Random Forest
El objetivo es implementar una función que:

Entrene un modelo Random Forest (RandomForestClassifier).

Haga predicciones en datos de prueba.

Evalúe el rendimiento del modelo con:

Precisión (accuracy_score)

Matriz de confusión (confusion_matrix)

Reporte de clasificación (classification_report)

Devuelva los resultados en un diccionario.

Supervise la implementación con pruebas unitarias (unittest).



Instrucciones

Implementa una función llamada entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test) que:

Entrene un RandomForestClassifier(n_estimators=100, random_state=42).

Prediga los valores de X_test.

Calcule las métricas de evaluación mencionadas.

Devuelva un diccionario con:

"predicciones": Array de predicciones del modelo.

"accuracy": Precisión del modelo en los datos de prueba.

"matriz_confusion": Matriz de confusión.

"reporte": Reporte de clasificación.

Usa el dataset de vinos (wine dataset) de sklearn.datasets.

Asegúrate de que el modelo tenga al menos 90% de precisión en los datos de prueba.



Ejemplo de Uso

El siguiente código debería ejecutarse correctamente una vez que implementes la función:

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
 
# Cargar el dataset de vinos
wine = load_wine()
X = wine.data  # Características
y = wine.target  # Clases de vinos
 
# Dividir en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Importar la función implementada
from solution import entrenar_y_evaluar_random_forest
 
# Llamar a la función y obtener las métricas
resultados = entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test)
 
# Mostrar los resultados
print("Precisión del modelo:", resultados["accuracy"])
print("Matriz de Confusión:\n", resultados["matriz_confusion"])
print("Reporte de Clasificación:\n", resultados["reporte"])




Salida esperada (aproximada)

Precisión del modelo: 1.0
Matriz de Confusión:
 [[14  0  0]
 [ 0 14  0]
 [ 0  0  8]]
Reporte de Clasificación:
               precision    recall  f1-score   support
 
     Clase 0       1.00      1.00      1.00        14
     Clase 1       1.00      1.00      1.00        14
     Clase 2       1.00      1.00      1.00         8
 
    accuracy                           1.00        36
   macro avg       1.00      1.00      1.00        36
weighted avg       1.00      1.00      1.00        36
