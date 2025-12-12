from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
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
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    resultados = entrenar_y_evaluar_svm(X_train, y_train, X_test, y_test)
    print("Precisión del modelo:", resultados["accuracy"])
    print("Matriz de Confusión:\n", resultados["matriz_confusion"])
    print("Reporte de Clasificación:\n", resultados["reporte"])