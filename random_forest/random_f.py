from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    predicciones = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)
    matriz_confusion = confusion_matrix(y_test, predicciones)
    reporte = classification_report(y_test, predicciones)
    return {
        "predicciones": predicciones,
        "accuracy": accuracy,
        "matriz_confusion": matriz_confusion,
        "reporte": reporte
    }

if __name__ == "__main__":
    # Ejemplo de uso con datos sintéticos
    X, y = make_classification(n_samples=100, n_features=4, n_informative=4, n_redundant=0,
                            n_classes=3, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=42, stratify=y)
    resultados = entrenar_y_evaluar_random_forest(X_train, y_train, X_test, y_test)
    print(f"Precisión del modelo: {resultados['accuracy']}")
    print("Matriz de Confusión:\n", resultados['matriz_confusion'])
    print("Reporte de Clasificación:\n", resultados['reporte'])
