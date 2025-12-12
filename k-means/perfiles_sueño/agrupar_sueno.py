import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SleepProfile:
    def __init__(self, duration, latency, wakeups, variability):
        self.duration = duration
        self.latency = latency
        self.wakeups = wakeups
        self.variability = variability
    def to_vector(self):
        return [self.duration, self.latency, self.wakeups, self.variability]

class SleepDatasetGenerator:
    def __init__(self, n=300):
        self.n = n
    def generate(self):
        np.random.seed(42)
        duration = np.random.normal(7, 1.2, self.n)
        latency = np.abs(np.random.normal(20, 10, self.n))
        wakeups = np.random.poisson(1.5, self.n)
        variability = np.abs(np.random.normal(30, 15, self.n))
        profiles = [SleepProfile(duration[i], latency[i], wakeups[i], variability[i]) for i in range(self.n)]
        return profiles

class SleepClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.kmeans = self.model
    def fit(self, profiles):
        X = np.array([p.to_vector() for p in profiles])
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        labels = self.model.labels_
        return X_scaled, labels
    def get_cluster_centers(self):
        centers_scaled = self.model.cluster_centers_
        return self.scaler.inverse_transform(centers_scaled)

class SleepAnalysisExample:
    def run(self):
        generator = SleepDatasetGenerator()
        profiles = generator.generate()
        clusterer = SleepClusterer(n_clusters=3)
        X_scaled, labels = clusterer.fit(profiles)
        centers = clusterer.get_cluster_centers()
        print("\n\U0001F4CC Centroides de los grupos:")
        for i, c in enumerate(centers):
            print(f"Grupo {i}: Duración={c[0]:.2f}h, Latencia={c[1]:.1f}min, Despertares={c[2]:.1f}, Variabilidad={c[3]:.1f}min")
        # Visualización
        df = pd.DataFrame({
            'Duración': [p.duration for p in profiles],
            'Latencia': [p.latency for p in profiles],
            'Despertares': [p.wakeups for p in profiles],
            'Variabilidad': [p.variability for p in profiles],
            'Cluster': labels
        })
        colors = ['red', 'green', 'blue']
        plt.figure(figsize=(8,6))
        for i in range(3):
            subset = df[df['Cluster'] == i]
            plt.scatter(subset['Duración'], subset['Variabilidad'], c=colors[i], label=f'Grupo {i}', alpha=0.6, edgecolors='k')
        plt.xlabel('Duración del sueño (h)')
        plt.ylabel('Variabilidad hora dormir (min)')
        plt.title('Agrupación de perfiles de sueño')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    example = SleepAnalysisExample()
    example.run()
