import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Clase Snack
class Snack:
    def __init__(self, calories, sugar, protein, fat, fiber, is_healthy=None):
        self.calories = calories
        self.sugar = sugar
        self.protein = protein
        self.fat = fat
        self.fiber = fiber
        self.is_healthy = is_healthy

    def to_vector(self):
        return [self.calories, self.sugar, self.protein, self.fat, self.fiber]

# Clase para generar datos sintéticos
class SnackGenerator:
    def __init__(self, n=100):
        self.n = n

    def generate(self):
        snacks = []
        np.random.seed(42)
        for _ in range(self.n):
            calories = np.random.randint(50, 501)
            sugar = np.random.randint(0, 51)
            protein = np.random.randint(0, 31)
            fat = np.random.randint(0, 31)
            fiber = np.random.randint(0, 16)
            is_healthy = int(
                calories < 200 and sugar < 15 and fat < 10 and (protein >= 5 or fiber >= 5)
            )
            snacks.append(Snack(calories, sugar, protein, fat, fiber, is_healthy))
        return snacks

# Clasificador usando árbol de decisión
class SnackClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)

    def fit(self, snacks):
        X = [s.to_vector() for s in snacks]
        y = [s.is_healthy for s in snacks]
        self.model.fit(X, y)

    def predict(self, snack):
        return self.model.predict([snack.to_vector()])[0]

# Ejemplo de uso
class SnackRecommendationExample:
    def run(self):
        generator = SnackGenerator(n=100)
        snacks = generator.generate()
        clf = SnackClassifier()
        clf.fit(snacks)
        # Snack de prueba
        test_snack = Snack(150, 10, 6, 5, 3)
        print("🔍 Snack Info:")
        print(f"Calories: {test_snack.calories}, Sugar: {test_snack.sugar}g, Protein: {test_snack.protein}g, Fat: {test_snack.fat}g, Fiber: {test_snack.fiber}g")
        pred = clf.predict(test_snack)
        if pred == 1:
            print("✅ Predicción: Este snack es saludable.")
        else:
            print("✅ Predicción: Este snack no es saludable.")

# Ejecutar ejemplo
if __name__ == "__main__":
    example = SnackRecommendationExample()
    example.run()