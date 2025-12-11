from sklearn.linear_model import LinearRegression

# --- Clase Player ---
class Player:
    def __init__(self, name, avg_session_time, avg_actions_per_min, avg_kills_per_session, victories=0):
        
            self.name = name
            self.avg_session_time = avg_session_time
            self.avg_actions_per_min = avg_actions_per_min
            self.avg_kills_per_session = avg_kills_per_session
            self.victories = victories

    def to_features(self):
        return [
            self.avg_session_time,
            self.avg_actions_per_min,
            self.avg_kills_per_session
            ]

# --- Clase PlayerDataset ---
class PlayerDataset:
    
    def __init__(self, players):
        self.players = players
        
    def get_feature_matrix(self):
        matrix = []
        for player in self.players:
            matrix.append(player.to_features())
        return matrix
        
    def get_target_vector(self):
        victory_list = []
        for player in self.players:
            if player.victories is not None:
                victory_list.append(player.victories)

# --- Clase VictoryPredictor ---
class VictoryPredictor:
    
    def __init__(self):
        self.model = LinearRegression()

    def train(self, dataset: PlayerDataset):
        X = dataset.get_feature_matrix()
        y = dataset.get_target_vector()
        self.model.fit(X, y)
        
    def predict(self, player):
        features = [player.to_features()]
        return self.model.predict(features)[0]


players = [
    Player("Alice", 40, 50, 6, 20),
    Player("Bob", 30, 35, 4, 10),
    Player("Charlie", 50, 60, 7, 25),
    Player("Diana", 20, 25, 2, 5),
    Player("Eve", 60, 70, 8, 30)
]
dataset = PlayerDataset(players)
predictor = VictoryPredictor()
predictor.train(dataset)

test_player = Player("TestPlayer", 45, 55, 5)
predicted = predictor.predict(test_player)
print(f"Victorias predichas para {test_player.name}: {predicted:.2f}")