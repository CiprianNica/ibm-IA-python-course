# Predicción para acertar la lotería con RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class GeneradorSeries:
	"""
	Genera combinaciones aleatorias de lotería (6 números únicos entre 1 y 49).
	"""
	def __init__(self, seed=42):
		self.seed = seed

	def generar_series(self, cantidad):
		np.random.seed(self.seed)
		series = []
		for _ in range(cantidad):
			combinacion = np.random.choice(np.arange(1, 50), size=6, replace=False)
			combinacion.sort()
			series.append(combinacion.tolist())
		return series

class DatosLoteria:
	"""
	Genera datos de entrenamiento con etiquetas de éxito (10% ganadoras).
	"""
	def __init__(self, seed=42):
		self.seed = seed

	def generar_datos_entrenamiento(self, cantidad=1000):
		gen = GeneradorSeries(seed=self.seed)
		series = gen.generar_series(cantidad)
		np.random.seed(self.seed)
		exito = np.random.choice([0, 1], size=cantidad, p=[0.9, 0.1])
		df = pd.DataFrame(series, columns=[f'N{i+1}' for i in range(6)])
		df['Exito'] = exito
		return df

class ModeloLoteria:
	"""
	Modelo RandomForest para predecir probabilidad de éxito de combinaciones.
	"""
	def __init__(self):
		self.scaler = StandardScaler()
		self.modelo = RandomForestClassifier(random_state=42)
		self.X_train = None
		self.y_train = None

	def entrenar(self, X, y):
		X_scaled = self.scaler.fit_transform(X)
		self.modelo.fit(X_scaled, y)
		self.X_train = X_scaled
		self.y_train = y

	def predecir_probabilidades(self, X):
		X_scaled = self.scaler.transform(X)
		probas = self.modelo.predict_proba(X_scaled)[:, 1]
		return probas

class VisualizadorResultados:
	"""
	Visualiza las top combinaciones con mayor probabilidad de éxito.
	"""
	def graficar_top_combinaciones(self, df_series, probabilidades, top_n=10):
		idx = np.argsort(probabilidades)[-top_n:][::-1]
		top_series = df_series.iloc[idx]
		top_probs = probabilidades[idx]
		etiquetas = [str(list(map(int, row))) for row in top_series.values]
		plt.figure(figsize=(10, 6))
		plt.barh(etiquetas[::-1], top_probs[::-1], color='skyblue')
		plt.xlabel('Probabilidad de éxito')
		plt.title(f'Top {top_n} combinaciones de lotería más prometedoras')
		plt.tight_layout()
		plt.show()

class EjecutarSimulacion:
	"""
	Ejecuta el flujo completo de simulación, entrenamiento, predicción y visualización.
	"""
	def ejecutar(self):
		# Generar datos de entrenamiento
		datos = DatosLoteria()
		df = datos.generar_datos_entrenamiento(1000)
		X = df[[f'N{i+1}' for i in range(6)]]
		y = df['Exito']
		# Entrenar modelo
		modelo = ModeloLoteria()
		modelo.entrenar(X, y)
		# Generar nuevas combinaciones
		gen = GeneradorSeries(seed=123)
		nuevas_series = gen.generar_series(100)
		df_nuevas = pd.DataFrame(nuevas_series, columns=[f'N{i+1}' for i in range(6)])
		# Predecir probabilidades
		probas = modelo.predecir_probabilidades(df_nuevas)
		# Encontrar la mejor combinación
		idx_max = np.argmax(probas)
		mejor_serie = df_nuevas.iloc[idx_max].values.astype(int)
		mejor_proba = probas[idx_max]
		print("\n🎯 Mejor serie encontrada:")
		print(f"Números: {list(mejor_serie)}")
		print(f"Probabilidad estimada de éxito: {mejor_proba:.4f}")
		# Visualizar top 10
		visual = VisualizadorResultados()
		visual.graficar_top_combinaciones(df_nuevas, probas, top_n=10)

if __name__ == "__main__":
	simulacion = EjecutarSimulacion()
	simulacion.ejecutar()
