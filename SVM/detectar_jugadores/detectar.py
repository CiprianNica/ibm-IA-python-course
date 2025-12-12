# Clase GameSimulator para pruebas automáticas
class GameSimulator:
    def run(self):
        X, y = simulate_players(100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = ProPlayerClassifier()
        clf.train(X_train, y_train)
        acc = clf.evaluate(X_test, y_test)
        print("Jugador profesional:")
        print("Precisión del modelo: {:.2f}".format(acc))
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class ProPlayerClassifier:
    def __init__(self):
        self.model = SVC()
    def train(self, X, y):
        self.model.fit(X, y)
    def predict(self, player_stats):
        return self.model.predict([player_stats])[0]
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)

# Simulador de datos de jugadores
def simulate_players(n=100, seed=42):
    np.random.seed(seed)
    X = np.random.rand(n, 5)
    # Regla: profesional si suma de stats > 3.2
    y = (X.sum(axis=1) > 3.2).astype(int)
    return X, y

if __name__ == "__main__":
    # Generar datos
    X, y = simulate_players(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = ProPlayerClassifier()
    clf.train(X_train, y_train)
    acc = clf.evaluate(X_test, y_test)
    print("Jugador profesional:")
    print("Precisión del modelo: {:.2f}".format(acc))
    # Prueba con dos jugadores
    player1 = np.array([0.9, 0.8, 0.7, 0.9, 0.8])  # profesional
    player2 = np.array([0.2, 0.3, 0.1, 0.4, 0.2])  # casual
    print("Jugador 1 predicción:", "Profesional" if clf.predict(player1) == 1 else "Casual")
    print("Jugador 2 predicción:", "Profesional" if clf.predict(player2) == 1 else "Casual")
