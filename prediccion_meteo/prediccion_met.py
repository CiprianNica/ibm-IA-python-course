import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Clase que representa un registro meteorológico
class WeatherRecord:
    def __init__(self, humidity, pressure, will_rain):
        self.humidity = humidity
        self.pressure = pressure
        self.will_rain = will_rain

    def to_vector(self):
        return [self.humidity, self.pressure]

# Generador de datos sintéticos
class WeatherDataGenerator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples

    def generate(self):
        records = []
        for _ in range(self.num_samples):
            humidity = np.random.uniform(30, 100)
            pressure = np.random.uniform(980, 1040)
            rain_prob = (humidity - 50) * 0.03 - (pressure - 1010) * 0.02
            rain_prob = 1 / (1 + np.exp(-rain_prob))
            will_rain = 1 if rain_prob >= 0.5 else 0
            records.append(WeatherRecord(humidity, pressure, will_rain))
        return records

# Clase que entrena el modelo de regresión logística
class WeatherRainClassifier:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, records):
        X = [r.to_vector() for r in records]
        y = [r.will_rain for r in records]
        self.model.fit(X, y)

    def predict(self, humidity, pressure):
        return self.model.predict([[humidity, pressure]])[0]

    def evaluate(self, records):
        X = [r.to_vector() for r in records]
        y_true = [r.will_rain for r in records]
        y_pred = self.model.predict(X)
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))

# Clase para ejecutar el ejemplo completo
class WeatherRainPredictionExample:
    def run(self):
        # Generar datos
        generator = WeatherDataGenerator(num_samples=200)
        records = generator.generate()
        
        # Dividir en entrenamiento y prueba
        train_data, test_data = train_test_split(records, test_size=0.3, random_state=42)
        
        # Entrenar modelo
        classifier = WeatherRainClassifier()
        classifier.fit(train_data)
        
        # Evaluar modelo
        classifier.evaluate(test_data)
        
        # Predicción para condiciones nuevas
        humidity_new = 80
        pressure_new = 995
        prediction = classifier.predict(humidity_new, pressure_new)
        print(f"\n🔍 Predicción para condiciones nuevas:")
        print(f"   Humedad: {humidity_new}%")
        print(f"   Presión: {pressure_new} hPa")
        print(f"   ¿Lloverá?: {'Sí ☔' if prediction == 1 else 'No ☀️'}")
        
        # Gráfico
        self.plot_results(test_data, classifier)
    
    def plot_results(self, records, classifier):
        humidity = [r.humidity for r in records]
        pressure = [r.pressure for r in records]
        will_rain = [r.will_rain for r in records]
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(humidity, pressure, c=will_rain, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='Lluvia (0=No, 1=Sí)')
        plt.xlabel('Humedad (%)')
        plt.ylabel('Presión (hPa)')
        plt.title('Predicción Meteorológica - Lluvia')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        
# Ejecutar ejemplo
example = WeatherRainPredictionExample()
example.run()
