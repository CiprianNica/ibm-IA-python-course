import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Función de clasificación KNN
def knn_clasificacion(datos, k=3):
    X = datos[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = datos['species']
    knn = KNeighborsClassifier(n_neighbors=k)
    return knn.fit(X, y)
    
# Cargar datos directamente desde sklearn
iris = load_iris(as_frame=True)
data = iris.frame
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Mapear números a nombres de especies
species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
data['species'] = data['species'].map(species_names)

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