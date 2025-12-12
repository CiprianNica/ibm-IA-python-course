import numpy as np
from sklearn.cluster import KMeans

class Traveler:
    def __init__(self, beach, mountain, city, countryside):
        self.beach = beach
        self.mountain = mountain
        self.city = city
        self.countryside = countryside
    def to_vector(self):
        return [self.beach, self.mountain, self.city, self.countryside]

class TravelerGenerator:
    def __init__(self, num_travelers=200):
        self.num_travelers = num_travelers
    def generate(self):
        np.random.seed(42)
        travelers = []
        for _ in range(self.num_travelers):
            beach = np.round(np.random.uniform(0, 10), 2)
            mountain = np.round(np.random.uniform(0, 10), 2)
            city = np.round(np.random.uniform(0, 10), 2)
            countryside = np.round(np.random.uniform(0, 10), 2)
            travelers.append(Traveler(beach, mountain, city, countryside))
        return travelers

class TravelerClusterer:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
    def fit(self, travelers):
        X = [t.to_vector() for t in travelers]
        self.model.fit(X)
    def predict(self, traveler):
        return int(self.model.predict([traveler.to_vector()])[0])
    def get_cluster_centers(self):
        return self.model.cluster_centers_

class TravelerClusteringExample:
    def run(self):
        generator = TravelerGenerator()
        travelers = generator.generate()
        clusterer = TravelerClusterer()
        clusterer.fit(travelers)
        centers = clusterer.get_cluster_centers()
        print("\U0001F3DD\U0001F3D4\U0001F3D9\U0001F304 Cluster Centers (Preferencias promedio):")
        for i, c in enumerate(centers):
            print(f"Cluster {i}: Playa={c[0]:.2f}, Montaña={c[1]:.2f}, Ciudad={c[2]:.2f}, Campo={c[3]:.2f}")
        new_traveler = Traveler(beach=9, mountain=2, city=8, countryside=1)
        cluster = clusterer.predict(new_traveler)
        print("\n\U0001F50D Nuevo viajero con preferencias:")
        print(f"Playa: {new_traveler.beach:.2f}, Montaña: {new_traveler.mountain:.2f}, Ciudad: {new_traveler.city:.2f}, Campo: {new_traveler.countryside:.2f}")
        print(f"\U0001F4CC El nuevo viajero pertenece al grupo {cluster}.")

if __name__ == "__main__":
    example = TravelerClusteringExample()
    example.run()
