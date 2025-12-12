from sklearn.neighbors import KNeighborsClassifier

class Player:
    def __init__(self, name, level, aggressiveness, cooperation, exploration, preferred_class=None):
        self.name = name
        self.level = level
        self.aggressiveness = aggressiveness
        self.cooperation = cooperation
        self.exploration = exploration
        self.preferred_class = preferred_class

    def to_features(self):
        return [self.level, self.aggressiveness, self.cooperation, self.exploration]

class PlayerDataset:
    def __init__(self, players):
        self.players = players

    def get_X(self):
        return [p.to_features() for p in self.players]

    def get_y(self):
        return [p.preferred_class for p in self.players]

class ClassRecommender:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.players = None

    def train(self, dataset: PlayerDataset):
        X = dataset.get_X()
        y = dataset.get_y()
        self.model.fit(X, y)
        self.players = dataset.players

    def predict(self, player: Player):
        return self.model.predict([player.to_features()])[0]

    def get_nearest_neighbors(self, player: Player):
        distances, indices = self.model.kneighbors([player.to_features()])
        return indices[0]

# Ejemplo de uso
if __name__ == "__main__":
    players = [
        Player("Alice", 20, 0.8, 0.2, 0.1, "Warrior"),
        Player("Bob", 45, 0.4, 0.8, 0.2, "Healer"),
        Player("Cleo", 33, 0.6, 0.4, 0.6, "Archer"),
        Player("Dan", 60, 0.3, 0.9, 0.3, "Healer"),
        Player("Eli", 50, 0.7, 0.2, 0.9, "Mage"),
        Player("Fay", 25, 0.9, 0.1, 0.2, "Warrior"),
    ]
    new_player = Player("TestPlayer", 40, 0.6, 0.3, 0.8)
    dataset = PlayerDataset(players)
    recommender = ClassRecommender(n_neighbors=3)
    recommender.train(dataset)
    recommended_class = recommender.predict(new_player)
    neighbors_indices = recommender.get_nearest_neighbors(new_player)
    print(f"Clase recomendada para {new_player.name}: {recommended_class}")
    print("Jugadores similares:")
    for i in neighbors_indices:
        print(f"- {players[i].name} ({players[i].preferred_class})")


