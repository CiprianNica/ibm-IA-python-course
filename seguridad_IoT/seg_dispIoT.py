import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import unittest

class IoTKNNClassifier:
    def __init__(self, n_neighbors=3, n_samples=50):
        """Inicializa el clasificador k-NN y genera datos sintéticos."""
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        # Generamos datos sintéticos
        np.random.seed(42)
        self.df = pd.DataFrame({
            'paquetes_por_segundo': np.random.randint(10, 1001, n_samples),
            'bytes_por_paquete': np.random.randint(50, 1501, n_samples),
            'protocolo': np.random.choice([1, 2, 3], n_samples),
            'seguro': np.random.choice([0, 1], n_samples)
        })
        # Dividimos en datos de entrenamiento y prueba
        self.X = self.df.drop(columns=["seguro"])
        self.y = self.df["seguro"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def train(self):
        """Entrena el modelo k-NN."""
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        """Evalúa el modelo y retorna la precisión."""
        y_pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, y_pred)

    def predict(self, nuevo_dispositivo):
        """Predice si un nuevo dispositivo IoT es seguro o peligroso."""
        return self.model.predict([nuevo_dispositivo])[0]


# Ejemplo de uso
if __name__ == "__main__":
    clasificador = IoTKNNClassifier(n_neighbors=3, n_samples=50)
    clasificador.train()
    print("Precisión del modelo:", clasificador.evaluate())
    # Nuevo dispositivo IoT
    nuevo = [300, 1000, 1]  # 300 paquetes/segundo, 1000 bytes/paquete, protocolo TCP
    resultado = clasificador.predict(nuevo)
    if resultado == 1:
        print("✅ Dispositivo seguro")
    else:
        print("⚠️ Dispositivo peligroso")


