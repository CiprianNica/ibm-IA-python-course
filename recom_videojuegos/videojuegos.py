import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Clase que representa un videojuego
class VideoGame:
    def __init__(self, action, strategy, graphics, difficulty, liked=None):
        self.action = action
        self.strategy = strategy
        self.graphics = graphics
        self.difficulty = difficulty
        self.liked = liked

    def to_vector(self):
        return [self.action, self.strategy, self.graphics, self.difficulty]

# 2. Clase para generar videojuegos aleatorios
class VideoGameGenerator:
    def __init__(self, n_games=100, seed=42):
        self.n_games = n_games
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        games = []
        for _ in range(self.n_games):
            action = np.round(np.random.uniform(0, 1), 2)
            strategy = np.round(np.random.uniform(0, 1), 2)
            graphics = np.round(np.random.uniform(0, 1), 2)
            difficulty = np.round(np.random.uniform(0, 1), 2)
            liked = int((action > 0.6 or graphics > 0.7) and difficulty < 0.7)
            games.append(VideoGame(action, strategy, graphics, difficulty, liked))
        return games

# 3. Clasificador usando Random Forest
class VideoGameClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def fit(self, games):
        X = [g.to_vector() for g in games]
        y = [g.liked for g in games]
        self.model.fit(X, y)

    def predict(self, game):
        pred = self.model.predict([game.to_vector()])[0]
        return pred

# 4. Ejemplo de uso
class VideoGameRecommendationExample:
    def run(self):
        generator = VideoGameGenerator()
        games = generator.generate()
        clf = VideoGameClassifier()
        clf.fit(games)
        new_game = VideoGame(action=0.9, strategy=0.4, graphics=0.8, difficulty=0.3)
        pred = clf.predict(new_game)
        print("🎮 Nuevo juego:")
        print(f"Action: {new_game.action}, Strategy: {new_game.strategy}, Graphics: {new_game.graphics}, Difficulty: {new_game.difficulty}")
        print(f"✅ Le gustará al jugador el juego? {'Si!' if pred == 1 else 'No'}")

# Ejecutar ejemplo
if __name__ == "__main__":
    example = VideoGameRecommendationExample()
    example.run()