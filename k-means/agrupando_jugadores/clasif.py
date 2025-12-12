from sklearn.cluster import KMeans

class Player:
    def __init__(self, name, avg_session_time, missions_completed, accuracy, aggressiveness):
        self.name = name
        self.avg_session_time = avg_session_time
        self.missions_completed = missions_completed
        self.accuracy = accuracy
        self.aggressiveness = aggressiveness
    def to_vector(self):
        return [self.avg_session_time, self.missions_completed, self.accuracy, self.aggressiveness]

class PlayerClusterer:
    def __init__(self):
        self.model = None
        self.players = None
        self.labels = None
    def fit(self, players, n_clusters):
        self.players = players
        X = [p.to_vector() for p in players]
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(X)
        self.labels = self.model.labels_
    def predict(self, player):
        return int(self.model.predict([player.to_vector()])[0])
    def get_cluster_centers(self):
        return self.model.cluster_centers_
    def print_cluster_summary(self, players):
        X = [p.to_vector() for p in players]
        labels = self.model.predict(X)
        clusters = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(players[idx].name)
        for cluster_id in sorted(clusters.keys()):
            print(f"Cluster {cluster_id}:")
            for name in clusters[cluster_id]:
                print(f"  - {name}")

class GameAnalytics:
    def run(self):
        data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            ("Diana", 0.8, 15, 0.55, 0.9),
            ("Eve", 2.7, 120, 0.88, 0.25),
            ("Frank", 1.1, 30, 0.62, 0.65),
            ("Grace", 0.9, 18, 0.58, 0.85),
            ("Hank", 3.2, 160, 0.91, 0.15)
        ]
        players = [Player(*row) for row in data]
        clusterer = PlayerClusterer()
        clusterer.fit(players, n_clusters=3)
        clusterer.print_cluster_summary(players)
        # Predicción para Zoe
        zoe = Player("Zoe", 1.5, 45, 0.65, 0.5)
        cluster = clusterer.predict(zoe)
        print(f"\nJugador Zoe pertenece al cluster: {cluster}")

if __name__ == "__main__":
    analytics = GameAnalytics()
    analytics.run()
