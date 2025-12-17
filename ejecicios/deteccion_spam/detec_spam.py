# Detección de correo electrónico spam con SVM
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generar_datos_emails(num_muestras):
	np.random.seed(42)
	longitud_mensaje = np.random.randint(50, 501, num_muestras)
	frecuencia_palabra_clave = np.random.uniform(0, 1, num_muestras)
	cantidad_enlaces = np.random.randint(0, 11, num_muestras)
	X = np.column_stack((longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces))
	# Etiqueta: Spam si frecuencia_palabra_clave > 0.6 o cantidad_enlaces > 5 o longitud_mensaje < 120
	y = ((frecuencia_palabra_clave > 0.6) | (cantidad_enlaces > 5) | (longitud_mensaje < 120)).astype(int)
	return X, y

def entrenar_modelo_svm(datos, etiquetas):
	X_train, X_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.3, random_state=42)
	modelo = SVC(probability=True, random_state=42)
	modelo.fit(X_train, y_train)
	y_pred = modelo.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Precisión del modelo SVM: {acc:.2f}")
	return modelo

def predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces):
	X_new = np.array([[longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces]])
	pred = modelo.predict(X_new)[0]
	if pred == 1:
		return "El email es Spam."
	else:
		return "El email NO es Spam."

# Ejemplo de uso
if __name__ == "__main__":
	X, y = generar_datos_emails(200)
	modelo = entrenar_modelo_svm(X, y)
	print(predecir_email(modelo, 80, 0.8, 7))
	print(predecir_email(modelo, 300, 0.2, 1))
