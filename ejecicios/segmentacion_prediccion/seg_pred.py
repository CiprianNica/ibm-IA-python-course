# Segmentación de Clientes y Predicción de Compra
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

class CustomerDataGenerator:
	"""
	Genera un DataFrame de clientes sintéticos con variables de compra y etiqueta de compra futura.
	"""
	def __init__(self, seed=42):
		self.seed = seed

	def generate(self, n=300):
		np.random.seed(self.seed)
		total_spent = np.random.uniform(50, 1500, n)
		total_purchases = np.random.randint(1, 51, n)
		purchase_frequency = np.random.uniform(0.5, 10, n)
		will_buy_next_month = ((total_spent > 500) & (purchase_frequency > 4)).astype(int)
		data = pd.DataFrame({
			'total_spent': total_spent,
			'total_purchases': total_purchases,
			'purchase_frequency': purchase_frequency,
			'will_buy_next_month': will_buy_next_month
		})
		return data

class CustomerSegmentationModel:
	"""
	Segmenta clientes y entrena un modelo de predicción de compra.
	"""
	def __init__(self, data):
		self.data = data.copy()
		self.model = None
		self.X_test = None
		self.y_test = None
		self.y_pred = None

	def segment_customers(self, n_clusters=3):
		kmeans = KMeans(n_clusters=n_clusters, random_state=42)
		features = self.data[['total_spent', 'total_purchases', 'purchase_frequency']]
		self.data['customer_segment'] = kmeans.fit_predict(features)

	def train_model(self):
		# Variables dummy para el segmento
		dummies = pd.get_dummies(self.data['customer_segment'], prefix='segment')
		X = pd.concat([
			self.data[['total_spent', 'total_purchases', 'purchase_frequency']],
			dummies
		], axis=1)
		y = self.data['will_buy_next_month']
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		self.model = LogisticRegression(max_iter=500, random_state=42)
		self.model.fit(X_train, y_train)
		self.X_test = X_test
		self.y_test = y_test
		self.y_pred = self.model.predict(X_test)

	def get_accuracy(self):
		return accuracy_score(self.y_test, self.y_pred)

	def get_confusion_matrix(self):
		return confusion_matrix(self.y_test, self.y_pred)

def graficar_segmentos(data):
	plt.figure(figsize=(8,6))
	for seg in sorted(data['customer_segment'].unique()):
		idx = data['customer_segment'] == seg
		plt.scatter(data.loc[idx, 'total_spent'], data.loc[idx, 'purchase_frequency'], label=f'Segmento {seg}', alpha=0.7)
	plt.xlabel('Total gastado (€)')
	plt.ylabel('Frecuencia de compra mensual')
	plt.title('Segmentación de clientes')
	plt.legend()
	plt.tight_layout()
	plt.show()

def graficar_probabilidad_compra(modelo):
	# Se asume que modelo es un LogisticRegression entrenado con variables dummy de segmento
	total_spent_range = np.linspace(50, 1500, 100)
	total_purchases = 25
	purchase_frequency = 5
	# Usar segmento 0 como referencia (todas las dummies en 0)
	X_pred = pd.DataFrame({
		'total_spent': total_spent_range,
		'total_purchases': total_purchases,
		'purchase_frequency': purchase_frequency,
		'segment_0': 0,
		'segment_1': 0,
		'segment_2': 0
	})
	# El modelo puede tener solo dos dummies si hay 3 segmentos, depende de pandas.get_dummies
	# Ajustar columnas según entrenamiento
	cols = modelo.feature_names_in_ if hasattr(modelo, 'feature_names_in_') else modelo.coef_.shape[1]
	for col in X_pred.columns:
		if col not in getattr(modelo, 'feature_names_in_', X_pred.columns):
			X_pred = X_pred.drop(columns=[col])
	proba = modelo.predict_proba(X_pred)[:,1]
	plt.figure(figsize=(8,6))
	plt.plot(total_spent_range, proba, color='blue')
	plt.xlabel('Total gastado (€)')
	plt.ylabel('Probabilidad de compra')
	plt.title('Probabilidad de compra vs. total gastado')
	plt.ylim(0,1)
	plt.grid(True)
	plt.tight_layout()
	plt.show()

# Ejemplo de uso
if __name__ == "__main__":
	# 1. Generar datos
	generador = CustomerDataGenerator()
	datos_clientes = generador.generate(300)
	# 2. Crear modelo
	modelo = CustomerSegmentationModel(datos_clientes)
	modelo.segment_customers()
	modelo.train_model()
	# 3. Resultados
	print("Precisión del modelo:", modelo.get_accuracy())
	print("Matriz de confusión:\n", modelo.get_confusion_matrix())
	# 4. Visualizaciones
	graficar_segmentos(modelo.data)
	graficar_probabilidad_compra(modelo.model)
