# Clasificación automática de frutas con KNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GeneradorFrutas:
	"""
	Genera muestras simuladas de frutas con peso, tamaño y etiqueta.
	"""
	def __init__(self, seed=42):
		self.seed = seed

	def generar(self, num_muestras):
		np.random.seed(self.seed)
		muestras = []
		etiquetas = []
		for _ in range(num_muestras):
			fruta = np.random.choice(['Manzana', 'Plátano', 'Naranja'])
			if fruta == 'Manzana':
				peso = np.random.uniform(120, 200)
				tam = np.random.uniform(7, 9)
			elif fruta == 'Plátano':
				peso = np.random.uniform(100, 150)
				tam = np.random.uniform(12, 20)
			else:  # Naranja
				peso = np.random.uniform(150, 250)
				tam = np.random.uniform(8, 12)
			muestras.append([peso, tam])
			etiquetas.append(fruta)
		return np.array(muestras), np.array(etiquetas)

class ClasificadorFrutas:
	"""
	Clasificador KNN para frutas.
	"""
	def __init__(self, k=5):
		self.k = k
		self.modelo = KNeighborsClassifier(n_neighbors=self.k)
		self.X_test = None
		self.y_test = None

	def entrenar(self, X, y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		self.modelo.fit(X_train, y_train)
		self.X_test = X_test
		self.y_test = y_test

	def evaluar(self):
		y_pred = self.modelo.predict(self.X_test)
		acc = accuracy_score(self.y_test, y_pred)
		print(f"\n🔍 Precisión del modelo: {acc*100:.2f}%")
		return acc

	def predecir(self, peso, tam):
		pred = self.modelo.predict(np.array([[peso, tam]]))[0]
		return pred

class VisualizadorFrutas:
	"""
	Visualiza los datos de frutas en un scatter plot.
	"""
	def graficar(self, X, y, titulo="Frutas"):
		colores = {'Manzana': 'red', 'Plátano': 'yellow', 'Naranja': 'orange'}
		plt.figure(figsize=(8,6))
		for fruta in np.unique(y):
			idx = y == fruta
			plt.scatter(X[idx,0], X[idx,1], c=colores[fruta], label=fruta, edgecolor='k', alpha=0.7)
		plt.xlabel('Peso (g)')
		plt.ylabel('Tamaño (cm)')
		plt.title(titulo)
		plt.legend()
		plt.tight_layout()
		plt.show()

class SimuladorFrutas:
	"""
	Ejecuta el flujo completo de generación, entrenamiento, predicción y visualización.
	"""
	def ejecutar(self):
		# Generar datos
		gen = GeneradorFrutas()
		X, y = gen.generar(100)
		# Entrenar clasificador
		clf = ClasificadorFrutas(k=5)
		clf.entrenar(X, y)
		clf.evaluar()
		# Predecir para nueva muestra
		peso_nuevo = 140
		tam_nuevo = 18
		pred = clf.predecir(peso_nuevo, tam_nuevo)
		print(f"🍎 La fruta predicha para peso={peso_nuevo}g y tamaño={tam_nuevo}cm es: {pred}")
		# Visualizar
		vis = VisualizadorFrutas()
		vis.graficar(X, y, titulo="Frutas generadas")

if __name__ == "__main__":
	simulador = SimuladorFrutas()
	simulador.ejecutar()
