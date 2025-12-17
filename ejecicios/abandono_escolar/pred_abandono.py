import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class SimuladorDatos:
    def __init__(self, n=200, seed=42):
        self.n = n
        self.seed = seed
    def generar(self):
        np.random.seed(self.seed)
        data = {
            'Edad': np.random.randint(18, 30, self.n),
            'Horas_estudio': np.random.uniform(0, 30, self.n),
            'Asistencia': np.random.uniform(50, 100, self.n),
            'Promedio': np.random.uniform(5, 10, self.n),
            'Uso_online': np.random.uniform(0, 15, self.n),
            'Abandono': np.random.choice([0, 1], size=self.n, p=[0.7, 0.3])
        }
        df = pd.DataFrame(data)
        return df

class ModeloAbandono:
    def __init__(self, max_depth=4, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.modelo = None
        self.X_test = None
        self.y_test = None
    def entrenar(self, data):
        X = data[['Edad', 'Horas_estudio', 'Asistencia', 'Promedio', 'Uso_online']]
        y = data['Abandono']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        self.modelo = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.modelo.fit(X_train, y_train)
        self.X_test = X_test
        self.y_test = y_test
        print("Entrenamiento completado.")
    def evaluar(self):
        y_pred = self.modelo.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print(f"\nPrecisión del modelo: {acc:.2f}")
        print("Reporte de clasificación:")
        print(classification_report(self.y_test, y_pred))
    def predecir_estudiante(self, estudiante_df):
        pred = self.modelo.predict(estudiante_df)[0]
        return "Abandonará" if pred == 1 else "Seguirá estudiando"

class TestBasicoModeloAbandono:
    def ejecutar(self):
        # Generar datos simulados
        sim = SimuladorDatos()
        df = sim.generar()
        print("Datos simulados:")
        print(df.head())
        # Entrenar modelo
        modelo = ModeloAbandono()
        modelo.entrenar(df)
        # Evaluar modelo
        modelo.evaluar()
        # Predicción para un nuevo estudiante
        nuevo_estudiante = pd.DataFrame({
            'Edad': [22],
            'Horas_estudio': [15],
            'Asistencia': [80],
            'Promedio': [7.5],
            'Uso_online': [5]
        })
        resultado = modelo.predecir_estudiante(nuevo_estudiante)
        print(f"\nEl estudiante probablemente: {resultado}")

if __name__ == "__main__":
    test = TestBasicoModeloAbandono()
    test.ejecutar()
