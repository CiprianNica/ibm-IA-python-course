# Predicción del precio de una vivienda usando regresión lineal
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class SimuladorViviendas:
	"""
	Genera un DataFrame sintético de viviendas con características y precio.
	"""
	def __init__(self, n=200, seed=42):
		self.n = n
		self.seed = seed

	def generar_datos(self):
		np.random.seed(self.seed)
		superficie = np.random.uniform(40, 200, self.n)
		habitaciones = np.random.randint(1, 6, self.n)
		antiguedad = np.random.randint(0, 51, self.n)
		distancia_centro = np.random.uniform(0.5, 25, self.n)
		banos = np.random.randint(1, 4, self.n)
		# Precio: función lineal + ruido
		precio = (
			superficie * 2500
			+ habitaciones * 20000
			- antiguedad * 1000
			- distancia_centro * 1500
			+ banos * 12000
			+ np.random.normal(0, 30000, self.n)
		)
		data = pd.DataFrame({
			'Superficie': superficie,
			'Habitaciones': habitaciones,
			'Antigüedad': antiguedad,
			'Distancia_centro': distancia_centro,
			'Baños': banos,
			'Precio': precio
		})
		return data

class ModeloPrecioVivienda:
	"""
	Modelo de regresión lineal para predecir el precio de una vivienda.
	"""
	def __init__(self):
		self.modelo = None
		self.X_test = None
		self.y_test = None

	def entrenar(self, data: pd.DataFrame):
		X = data[['Superficie', 'Habitaciones', 'Antigüedad', 'Distancia_centro', 'Baños']]
		y = data['Precio']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		self.modelo = LinearRegression()
		self.modelo.fit(X_train, y_train)
		self.X_test = X_test
		self.y_test = y_test
		print("Modelo entrenado correctamente.")

	def evaluar(self):
		y_pred = self.modelo.predict(self.X_test)
		mse = mean_squared_error(self.y_test, y_pred)
		r2 = r2_score(self.y_test, y_pred)
		print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
		print(f"R² del modelo: {r2:.2f}")

	def predecir(self, nueva_vivienda: pd.DataFrame) -> float:
		pred = self.modelo.predict(nueva_vivienda)[0]
		return pred

class TestModeloPrecio:
	"""
	Prueba integral del sistema de predicción de precio de vivienda.
	"""
	def ejecutar(self):
		# Generar datos
		sim = SimuladorViviendas()
		df = sim.generar_datos()
		print("Primeras filas de datos simulados:")
		print(df.head())
		# Entrenar y evaluar modelo
		modelo = ModeloPrecioVivienda()
		modelo.entrenar(df)
		modelo.evaluar()
		# Crear vivienda de ejemplo
		ejemplo = pd.DataFrame({
			'Superficie': [120],
			'Habitaciones': [3],
			'Antigüedad': [10],
			'Distancia_centro': [5],
			'Baños': [2]
		})
		precio_estimado = modelo.predecir(ejemplo)
		print(f"\nEl precio estimado de la vivienda es: ${precio_estimado:,.2f}")

if __name__ == "__main__":
	test = TestModeloPrecio()
	test.ejecutar()
