import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

class Piece:
    def __init__(self, texture, symmetry, edges, center_offset, label):
        self.texture = texture
        self.symmetry = symmetry
        self.edges = edges
        self.center_offset = center_offset
        self.label = label
    def to_vector(self):
        return [self.texture, self.symmetry, self.edges, self.center_offset]

class PieceDatasetGenerator:
    def __init__(self, n=400):
        self.n = n
    def generate(self):
        np.random.seed(42)
        pieces = []
        for _ in range(self.n):
            texture = np.clip(np.random.normal(0.6, 0.2), 0, 1)
            symmetry = np.clip(np.random.normal(0.7, 0.2), 0, 1)
            edges = np.clip(np.random.normal(40, 10), 10, 60)
            center_offset = np.clip(np.random.normal(0.15, 0.1), 0, 0.5)
            # Reglas de clasificación
            if (symmetry < 0.4 and center_offset > 0.25) or texture < 0.35 or edges < 30 or center_offset > 0.35:
                label = "Defectuosa"
            else:
                label = "Correcta"
            pieces.append(Piece(texture, symmetry, edges, center_offset, label))
        return pieces

class PieceClassifier:
    def __init__(self):
        self.model = SVC(kernel='rbf', gamma='scale', C=1.0)
    def fit(self, pieces: list) -> None:
        X = [p.to_vector() for p in pieces]
        y = [p.label for p in pieces]
        self.model.fit(X, y)
    def predict(self, texture, symmetry, edges, center_offset) -> str:
        pred = self.model.predict([[texture, symmetry, edges, center_offset]])[0]
        return pred
    def evaluate(self, test_data: list) -> None:
        X = [p.to_vector() for p in test_data]
        y_true = [p.label for p in test_data]
        y_pred = self.model.predict(X)
        print("\n📊 Matriz de confusión:")
        print(confusion_matrix(y_true, y_pred, labels=["Correcta", "Defectuosa"]))
        print("\n📝 Informe de clasificación:")
        print(classification_report(y_true, y_pred, labels=["Correcta", "Defectuosa"]))

class PieceAnalysisExample:
    def run(self) -> None:
        generator = PieceDatasetGenerator()
        pieces = generator.generate()
        train, test = train_test_split(pieces, test_size=0.3, random_state=42)
        clf = PieceClassifier()
        clf.fit(train)
        clf.evaluate(test)
        # Predicción personalizada
        t, s, e, o = 0.45, 0.5, 45, 0.15
        pred = clf.predict(t, s, e, o)
        print("\n🔎 Predicción de pieza personalizada:")
        print(f"  → Textura: {t:.2f}, Simetría: {s:.2f}, Bordes: {e:.0f}, Offset: {o:.2f}")
        print(f"  → Clasificación: {pred}")
        # Visualización
        df = pd.DataFrame({
            "Textura": [p.texture for p in pieces],
            "Simetría": [p.symmetry for p in pieces],
            "Bordes": [p.edges for p in pieces],
            "Offset": [p.center_offset for p in pieces],
            "Etiqueta": [p.label for p in pieces]
        })
        color_map = {"Correcta": "green", "Defectuosa": "red"}
        plt.figure(figsize=(8,6))
        for label, color in color_map.items():
            subset = df[df["Etiqueta"] == label]
            plt.scatter(subset["Textura"], subset["Offset"], c=color, label=label, alpha=0.7, edgecolors='k')
        plt.xlabel("Textura")
        plt.ylabel("Offset")
        plt.title("🏭 Clasificación de piezas industriales")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    example = PieceAnalysisExample()
    example.run()
