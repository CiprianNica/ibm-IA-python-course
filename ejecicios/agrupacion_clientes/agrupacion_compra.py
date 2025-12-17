# Agrupación de clientes según comportamientos de compra
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SimuladorClientes:
	"""
	Simula datos de clientes con monto gastado, frecuencia de compras y categorías preferidas.
	"""
	def __init__(self, n=200, seed=42):
		self.n = n
		self.seed = seed

	def generar_datos(self) -> np.ndarray:
		np.random.seed(self.seed)
		monto_gastado = np.random.uniform(100, 10000, self.n)
		frecuencia = np.random.randint(1, 101, self.n)
		categorias = np.random.randint(1, 6, (self.n, 3))
		total_categorias = categorias.sum(axis=1)
		datos = np.column_stack((monto_gastado, frecuencia, total_categorias))
		return datos

class ModeloSegmentacionClientes:
	"""
	Modelo de clustering KMeans para segmentar clientes.
	"""
	def __init__(self, n_clusters=3):
		self.n_clusters = n_clusters
		self.scaler = StandardScaler()
		self.modelo = KMeans(n_clusters=self.n_clusters, random_state=42)
		self.datos_escalados = None

	def entrenar(self, datos: np.ndarray) -> None:
		self.datos_escalados = self.scaler.fit_transform(datos)
		self.modelo.fit(self.datos_escalados)
		print(f"Modelo entrenado con {self.n_clusters} clusters.")

	def predecir(self, cliente_nuevo: list) -> int:
		cliente_arr = np.array(cliente_nuevo).reshape(1, -1)
		cliente_esc = self.scaler.transform(cliente_arr)
		cluster = self.modelo.predict(cliente_esc)[0]
		return cluster

class TestSegmentacionClientes:
	"""
	Prueba integral del sistema de segmentación de clientes.
	"""
	def ejecutar(self) -> None:
		# Generar datos
		sim = SimuladorClientes()
		datos = sim.generar_datos()
		print("Primeros 5 registros de datos simulados:")
		print(datos[:5])
		# Entrenar modelo
		modelo = ModeloSegmentacionClientes(n_clusters=3)
		modelo.entrenar(datos)
		# Predecir cluster para nuevo cliente
		cliente_nuevo = [2000, 10, 12]
		cluster = modelo.predecir(cliente_nuevo)
		print(f"El nuevo cliente pertenece al cluster: {cluster}")
		# Visualización
		plt.figure(figsize=(8,6))
		scatter = plt.scatter(
			datos[:,0], datos[:,1], c=modelo.modelo.labels_, cmap='viridis', alpha=0.7
		)
		plt.xlabel('Monto gastado')
		plt.ylabel('Frecuencia de compras')
		plt.title('Segmentación de clientes por KMeans')
		cbar = plt.colorbar(scatter)
		cbar.set_label('Cluster')
		plt.show()

if __name__ == "__main__":
	test = TestSegmentacionClientes()
	test.ejecutar()
