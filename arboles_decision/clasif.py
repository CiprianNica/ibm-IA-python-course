from sklearn.tree import DecisionTreeClassifier
import numpy as np

def entrenar_arbol_decision(X_train, y_train, X_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)

# Ejemplo de uso
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predicciones = entrenar_arbol_decision(X_train, y_train, X_test)
    print("Predicciones:", predicciones)
