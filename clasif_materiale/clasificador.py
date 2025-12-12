import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Clase que representa un registro de reciclaje
class RecyclableItem:
    def __init__(self, weight, volume, material_type):
        self.weight = weight
        self.volume = volume
        self.material_type = material_type

    def to_vector(self):
        return [self.weight, self.volume]

# Generador de datos sintéticos
class RecyclingDataGenerator:
    def __init__(self, num_samples=150):
        self.num_samples = num_samples

    def generate(self):
        items = []
        n = self.num_samples // 3
        # Papel
        for _ in range(n):
            w = np.random.normal(30, 5)
            v = np.random.normal(250, 30)
            items.append(RecyclableItem(w, v, 0))
        # Plástico
        for _ in range(n):
            w = np.random.normal(80, 10)
            v = np.random.normal(150, 20)
            items.append(RecyclableItem(w, v, 1))
        # Metal
        for _ in range(self.num_samples - 2*n):
            w = np.random.normal(150, 20)
            v = np.random.normal(80, 10)
            items.append(RecyclableItem(w, v, 2))
        return items

# Clasificador usando KNN
class RecyclableMaterialClassifier:
    def __init__(self, k=5):
        self.k = k
        self.model = KNeighborsClassifier(n_neighbors=k)

    def fit(self, records):
        X = [r.to_vector() for r in records]
        y = [r.material_type for r in records]
        self.model.fit(X, y)

    def predict(self, weight, volume):
        return self.model.predict([[weight, volume]])[0]

    def evaluate(self, records):
        X = [r.to_vector() for r in records]
        y_true = [r.material_type for r in records]
        y_pred = self.model.predict(X)
        print(confusion_matrix(y_true, y_pred))
        print(classification_report(y_true, y_pred))

# Clase de ejecución del ejemplo
class RecyclablePredictionExample:
    def run(self):
        # Generar datos
        generator = RecyclingDataGenerator(num_samples=150)
        items = generator.generate()
        # Separar en entrenamiento y prueba
        train, test = train_test_split(items, test_size=0.3, random_state=42)
        # Entrenar clasificador
        clf = RecyclableMaterialClassifier(k=5)
        clf.fit(train)
        # Evaluar
        clf.evaluate(test)
        # Predicción para nuevo objeto
        weight_new, volume_new = 60, 180
        pred = clf.predict(weight_new, volume_new)
        tipos = {0: 'Papel', 1: 'Plástico', 2: 'Metal'}
        print(f"\n📦 Predicción para un nuevo objeto:")
        print(f"   Peso: {weight_new}g, Volumen: {volume_new}cm³")
        print(f"   Tipo estimado: {tipos[pred]}")
        # Visualización
        self.plot_results(test, clf)

    def plot_results(self, records, clf):
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        labels = ['Papel', 'Plástico', 'Metal']
        for i in range(3):
            xs = [r.weight for r in records if r.material_type == i]
            ys = [r.volume for r in records if r.material_type == i]
            plt.scatter(xs, ys, c=colors[i], label=labels[i], alpha=0.7)
        plt.xlabel('Peso (g)')
        plt.ylabel('Volumen (cm³)')
        plt.title('Clasificación de Materiales Reciclables')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Ejecutar
if __name__ == "__main__":
    example = RecyclablePredictionExample()
    example.run()