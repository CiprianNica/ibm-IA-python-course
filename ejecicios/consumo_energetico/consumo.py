# Predicción del consumo energético con regresión lineal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class EnergyRecord:
	def __init__(self, temperature, consumption):
		self.temperature = temperature
		self.consumption = consumption
	def to_vector(self):
		return [self.temperature]

class EnergyDataGenerator:
	def __init__(self, n=100, seed=42):
		self.n = n
		self.seed = seed
	def generate(self):
		np.random.seed(self.seed)
		temps = np.random.uniform(-5, 35, self.n)
		noise = np.random.normal(0, 5, self.n)
		consumptions = 100 + (np.abs(temps - 20) * 3) + noise
		return [EnergyRecord(t, c) for t, c in zip(temps, consumptions)]

class EnergyRegressor:
	def __init__(self):
		self.model = LinearRegression()
	def fit(self, records):
		X = np.array([r.to_vector() for r in records])
		y = np.array([r.consumption for r in records])
		self.model.fit(X, y)
	def predict(self, temperature):
		return self.model.predict(np.array([[temperature]]))[0]
	def get_model(self):
		return self.model

class EnergyPredictionExample:
	def run(self):
		# Generar datos
		generator = EnergyDataGenerator()
		records = generator.generate()
		# Entrenar modelo
		reg = EnergyRegressor()
		reg.fit(records)
		# Predicción para nueva temperatura
		temp_pred = 30
		consumo_pred = reg.predict(temp_pred)
		print(f"🔍 Temperatura: {temp_pred} °C")
		print(f"⚡ Predicción de consumo: {consumo_pred:.2f} kWh")
		# Visualización
		temps = np.array([r.temperature for r in records])
		consumos = np.array([r.consumption for r in records])
		plt.figure(figsize=(8,6))
		plt.scatter(temps, consumos, label='Datos reales', alpha=0.7)
		# Línea de regresión
		x_line = np.linspace(-5, 35, 100)
		y_line = reg.get_model().predict(x_line.reshape(-1,1))
		plt.plot(x_line, y_line, color='red', label='Regresión lineal')
		plt.xlabel('Temperatura (°C)')
		plt.ylabel('Consumo energético (kWh)')
		plt.title('Consumo energético vs Temperatura')
		plt.legend()
		plt.tight_layout()
		plt.show()

# Ejemplo de uso
if __name__ == "__main__":
	example = EnergyPredictionExample()
	example.run()
