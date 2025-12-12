import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Clase que representa una persona con medidas fisiológicas
class Individual:
    def __init__(self, heart_rate, cortisol_level, skin_conductance, stress_level):
        self.heart_rate = heart_rate
        self.cortisol_level = cortisol_level
        self.skin_conductance = skin_conductance
        self.stress_level = stress_level

    def to_vector(self):
        return [self.heart_rate, self.cortisol_level, self.skin_conductance]

# Generador de datos simulados
class StressDataGenerator:
    def __init__(self, n_individuals=300, seed=42):
        self.n_individuals = n_individuals
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        individuals = []
        for _ in range(self.n_individuals):
            hr = np.round(np.random.normal(75, 15), 2)
            cort = np.round(np.random.normal(12, 4), 2)
            cond = np.round(np.random.normal(5, 1.5), 2)
            if hr > 90 or cort > 18 or cond > 6.5:
                level = 'Alto'
            elif hr > 70 or cort > 10 or cond > 4.5:
                level = 'Moderado'
            else:
                level = 'Bajo'
            individuals.append(Individual(hr, cort, cond, level))
        return individuals

# Clasificador con Random Forest
class StressClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.label_map = {'Bajo': 0, 'Moderado': 1, 'Alto': 2}
        self.inv_label_map = {0: 'Bajo', 1: 'Moderado', 2: 'Alto'}

    def fit(self, individuals):
        X = [ind.to_vector() for ind in individuals]
        y = [self.label_map[ind.stress_level] for ind in individuals]
        self.model.fit(X, y)

    def predict(self, heart_rate, cortisol, conductance):
        pred = self.model.predict([[heart_rate, cortisol, conductance]])[0]
        return self.inv_label_map[pred]

    def evaluate(self, test_data):
        X = [ind.to_vector() for ind in test_data]
        y_true = [self.label_map[ind.stress_level] for ind in test_data]
        y_pred = self.model.predict(X)
        labels = [0, 1, 2]
        print("\n📊 Matriz de confusión:")
        print(confusion_matrix(y_true, y_pred, labels=labels))
        print("\n📝 Informe de clasificación:")
        print(classification_report(y_true, y_pred, labels=labels, target_names=['Bajo', 'Moderado', 'Alto']))

# Ejecución y visualización
class StressAnalysisExample:
    def run(self):
        generator = StressDataGenerator()
        individuals = generator.generate()
        train, test = train_test_split(individuals, test_size=0.3, random_state=42)
        clf = StressClassifier()
        clf.fit(train)
        clf.evaluate(test)
        # Predicción personalizada
        hr, cort, cond = 95, 20, 7
        pred = clf.predict(hr, cort, cond)
        print("\n🧠 Predicción para individuo personalizado:")
        print(f"  Ritmo cardíaco: {hr}, Cortisol: {cort}, Conductancia: {cond}")
        print(f"  → Nivel estimado de estrés: {pred}")
        # Visualización
        df = pd.DataFrame({
            'heart_rate': [ind.heart_rate for ind in individuals],
            'cortisol': [ind.cortisol_level for ind in individuals],
            'conductance': [ind.skin_conductance for ind in individuals],
            'stress_level': [ind.stress_level for ind in individuals]
        })
        color_map = {'Bajo': 'green', 'Moderado': 'orange', 'Alto': 'red'}
        plt.figure(figsize=(8,6))
        for level, color in color_map.items():
            subset = df[df['stress_level'] == level]
            plt.scatter(subset['cortisol'], subset['heart_rate'], c=color, label=level, alpha=0.7, edgecolors='k')
        plt.xlabel('Nivel de cortisol (µg/dL)')
        plt.ylabel('Ritmo cardíaco (ppm)')
        plt.title('Niveles de estrés: cortisol vs ritmo cardíaco')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Ejecutar ejemplo
if __name__ == "__main__":
    example = StressAnalysisExample()
    example.run()
