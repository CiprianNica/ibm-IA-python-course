# Predicción de compra de un producto en línea
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generar_datos_compras(num_muestras):
	np.random.seed(42)
	num_paginas_vistas = np.random.randint(1, 21, num_muestras)
	tiempo_en_sitio = np.random.uniform(0, 30, num_muestras)
	datos = np.column_stack((num_paginas_vistas, tiempo_en_sitio))
	etiquetas = np.where((num_paginas_vistas > 5) & (tiempo_en_sitio > 10), 1, 0)
	return datos, etiquetas

def graficar_datos(datos, etiquetas):
	plt.figure(figsize=(8,6))
	compran = etiquetas == 1
	no_compran = etiquetas == 0
	plt.scatter(datos[compran,0], datos[compran,1], c='green', label='Compró', alpha=0.7)
	plt.scatter(datos[no_compran,0], datos[no_compran,1], c='red', label='No compró', alpha=0.7)
	plt.xlabel('Número de páginas vistas')
	plt.ylabel('Tiempo en el sitio (min)')
	plt.title('Comportamiento de usuarios y decisión de compra')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.show()

def entrenar_modelo(datos, etiquetas):
	X_train, X_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.3, random_state=42)
	modelo = LogisticRegression()
	modelo.fit(X_train, y_train)
	print(f"Precisión del modelo: {modelo.score(X_train, y_train):.2f}")
	return modelo

def predecir_compra(modelo, num_paginas_vistas, tiempo_en_sitio):
	pred = modelo.predict(np.array([[num_paginas_vistas, tiempo_en_sitio]]))[0]
	if pred == 1:
		return "El usuario comprará el producto."
	else:
		return "El usuario no comprará el producto."

def evaluar_modelo(modelo, datos, etiquetas):
	X_train, X_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.3, random_state=42)
	y_pred = modelo.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Precisión en el conjunto de prueba: {acc:.2f}")
	return acc

def graficar_funcion_prediccion(modelo):
	paginas = np.arange(1, 21)
	tiempo_fijo = 15
	X_pred = np.column_stack((paginas, np.full_like(paginas, tiempo_fijo)))
	proba = modelo.predict_proba(X_pred)[:,1]
	plt.figure(figsize=(8,6))
	plt.plot(paginas, proba, marker='o', color='blue')
	plt.xlabel('Número de páginas vistas')
	plt.ylabel('Probabilidad de compra')
	plt.title('Probabilidad de compra vs. páginas vistas (tiempo fijo = 15 min)')
	plt.ylim(0,1)
	plt.grid(True)
	plt.tight_layout()
	plt.show()

# Ejemplo de uso
if __name__ == "__main__":
	# Generamos los datos
	datos, etiquetas = generar_datos_compras(100)
	# Visualizamos los datos
	graficar_datos(datos, etiquetas)
	# Entrenamos el modelo
	modelo = entrenar_modelo(datos, etiquetas)
	# Evaluamos el modelo
	evaluar_modelo(modelo, datos, etiquetas)
	# Predicción de un nuevo usuario
	print(predecir_compra(modelo, 8, 12))
	# Graficar función de predicción
	graficar_funcion_prediccion(modelo)
