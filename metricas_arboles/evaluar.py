from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predicciones = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicciones)
    matriz_conf = confusion_matrix(y_test, predicciones)
    target_names = ['Setosa', 'Versicolor', 'Virginica']
    reporte = classification_report(y_test, predicciones, target_names=target_names)
    return {
        'predicciones': predicciones,
        'accuracy': accuracy,
        'matriz_confusion': matriz_conf,
        'reporte': reporte
    }

# Ejemplo de uso
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    resultados = entrenar_y_evaluar_arbol(X_train, y_train, X_test, y_test)
    print(f"Precisión del modelo: {resultados['accuracy']}")
    print("Matriz de Confusión:\n", resultados['matriz_confusion'])
    print("Reporte de Clasificación:\n", resultados['reporte'])
