📝 Enunciado

En este ejercicio desarrollarás una función llamada knn_clasificacion que implemente el algoritmo de los k vecinos más cercanos (KNN) para resolver un problema de clasificación.

Imagina que tienes un conjunto de datos que describe diferentes tipos de flores.

Cada flor tiene medidas de sus sépalos y pétalos (longitud y ancho), y queremos predecir la especie a la que pertenece basándonos en esas características.

Trabajaremos con el conjunto de datos Iris, que es ampliamente utilizado en el aprendizaje automático. Contiene observaciones de tres especies de flores: setosa, versicolor y virginica.



🎯 Objetivo

Construir una función llamada:

def knn_clasificacion(datos, k=3):

Esta función debe:

Recibir un DataFrame de pandas que contenga las siguientes columnas:

'sepal_length', 'sepal_width', 'petal_length', 'petal_width' (características numéricas)

'species' (etiqueta de clase)

Entrenar un modelo de clasificación KNN usando scikit-learn.

Devolver el modelo entrenado.



🔽 Datos disponibles

Los datos se obtienen directamente desde scikit-learn con el siguiente código:

from sklearn.datasets import load_iris
 
iris = load_iris(as_frame=True)
data = iris.frame
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']


🚀 Ejemplo de uso

# Cargar datos directamente desde sklearn
iris = load_iris(as_frame=True)
data = iris.frame
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
 
# Entrenar el modelo
modelo_knn = knn_clasificacion(data, k=3)
 
# Nuevas muestras para clasificar
nuevas_muestras = pd.DataFrame({
    'sepal_length': [5.1, 6.0, 4.4],
    'sepal_width': [3.5, 2.9, 3.2],
    'petal_length': [1.4, 4.5, 1.3],
    'petal_width': [0.2, 1.5, 0.2]
})
 
# Predicciones
estimaciones_clasificacion = modelo_knn.predict(nuevas_muestras)
print("Estimaciones de Clasificación:")
print(estimaciones_clasificacion)


Resultado esperado

Estimaciones de Clasificación:
['setosa' 'versicolor' 'setosa']