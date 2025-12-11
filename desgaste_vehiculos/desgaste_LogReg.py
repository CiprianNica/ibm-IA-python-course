import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Clase que representa un registro de uso de un vehículo militar
class VehicleRecord:
    def __init__(self, hours_used, wear_level):
        self.hours_used = hours_used
        self.wear_level = wear_level

    def to_vector(self):
        return [self.hours_used]

# Clase para generar datos sintéticos
class VehicleDataGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def generate(self):
        hours = np.random.uniform(50, 500, self.num_samples)
        wear_level = 10 + 0.18 * hours + np.random.normal(0, 5, self.num_samples)
        wear_level = np.clip(wear_level, 0, 100)
        records = [VehicleRecord(h, w) for h, w in zip(hours, wear_level)]
        return records

# Clase que entrena el modelo de regresión lineal
class VehicleWearRegressor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, records):
        X = np.array([r.to_vector() for r in records])
        y = np.array([r.wear_level for r in records])
        self.model.fit(X, y)

    def predict(self, hours):
        hours_array = np.array(hours).reshape(-1, 1)
        return self.model.predict(hours_array)

    def get_model(self):
        return self.model

# Clase principal para ejecutar el ejemplo
class VehicleWearPredictionExample:
    def run(self):
        # Generar datos
        generator = VehicleDataGenerator(num_samples=100)
        records = generator.generate()
        
        # Entrenar modelo
        regressor = VehicleWearRegressor()
        regressor.fit(records)
        
        # Predecir para 250 horas
        predicted_wear = regressor.predict([250])[0]
        print(f"Predicción para 250 horas: {predicted_wear:.2f}% de desgaste")
        
        # Visualizar
        self.plot_results(records, regressor)
    
    def plot_results(self, records, regressor):
        hours = np.array([r.hours_used for r in records])
        wear = np.array([r.wear_level for r in records])
        
        plt.figure(figsize=(10, 6))
        
        # Puntos verdes → Datos reales
        plt.scatter(hours, wear, color='green', alpha=0.6, label='Datos reales')
        
        # Línea roja → Línea de regresión
        hours_line = np.linspace(50, 500, 100)
        wear_pred = regressor.predict(hours_line)
        plt.plot(hours_line, wear_pred, 'r-', linewidth=2, label='Línea de regresión')
        
        # Línea gris vertical → Nuevo vehículo
        plt.axvline(x=250, color='gray', linestyle='--', linewidth=2, label='Nuevo vehículo (250h)')
        
        plt.xlabel('Horas de uso')
        plt.ylabel('Nivel de desgaste (%)')
        plt.title('Predicción de Desgaste de Vehículos Militares')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Ejecutar el ejemplo
if __name__ == "__main__":
    example = VehicleWearPredictionExample()
    example.run()
