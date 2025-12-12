import numpy as np
from sklearn.ensemble import RandomForestClassifier

class LanguagePredictor:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.label_map = {
            0: "Python",
            1: "JavaScript",
            2: "Java",
            3: "C++"
        }

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        pred = self.model.predict([features])[0]
        return self.label_map[pred]

def generate_dataset(n_samples=100, seed=42):
    np.random.seed(seed)
    X = np.zeros((n_samples, 5))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        velocidad = np.random.rand()
        mantenimiento = np.random.rand()
        libs = np.random.rand()
        tipo_app = np.random.randint(0, 3)
        rendimiento = np.random.rand()
        X[i] = [velocidad, mantenimiento, libs, tipo_app, rendimiento]
        if rendimiento > 0.8 and tipo_app == 2:
            lenguaje = 3  # C++
        elif mantenimiento > 0.7 and tipo_app == 1:
            lenguaje = 2  # Java
        elif libs > 0.6 and tipo_app == 0:
            lenguaje = 0  # Python
        else:
            lenguaje = 1  # JavaScript
        y[i] = lenguaje
    return X, y

if __name__ == "__main__":
    # Ejemplo de uso
    X, y = generate_dataset()
    predictor = LanguagePredictor()
    predictor.train(X, y)
    new_project = np.array([0.7, 0.9, 0.5, 1, 0.6])
    pred = predictor.predict(new_project)
    print(f"Lenguaje recomendado para el nuevo proyecto: {pred}")

