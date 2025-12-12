from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

def entrenar_y_evaluar_kmeans(X, y, k):
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(X)
    clusters = modelo.labels_
    inertia = modelo.inertia_
    silhouette = silhouette_score(X, clusters)
    rand_score = adjusted_rand_score(y, clusters)
    return {
        "clusters": clusters,
        "inertia": inertia,
        "silhouette_score": silhouette,
        "adjusted_rand_score": rand_score
    }

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    y = iris.target
    resultados = entrenar_y_evaluar_kmeans(X, y, k=3)
    print("Inercia del modelo:", resultados["inertia"])
    print("Silhouette Score:", resultados["silhouette_score"])
    print("Adjusted Rand Score:", resultados["adjusted_rand_score"])
    print("Clusters asignados:\n", resultados["clusters"][:10])
