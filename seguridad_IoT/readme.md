Seguridad en Dispositivos IoT
Contexto
Imagina que formas parte del equipo de defensa digital de un hogar inteligente. Con cada vez más dispositivos conectados a internet, se vuelve crucial detectar cuáles son seguros y cuáles podrían representar una amenaza.

Tu tarea será construir un modelo que aprenda a clasificar dispositivos IoT (Internet de las Cosas) como seguros (1) o peligrosos (0) según su tráfico de red, utilizando el algoritmo k-Nearest Neighbors (KNN).



🧠 Instrucciones

Crea una clase llamada IoTKNNClassifier que simule un sistema de detección para dispositivos IoT basado en aprendizaje automático.



🔧 1. Constructor: __init__(self, n_neighbors=3, n_samples=50)

Genera un conjunto de datos sintéticos de tamaño n_samples que simulen tráfico de red para varios dispositivos.

Cada muestra debe tener las siguientes características:

paquetes_por_segundo: un entero entre 10 y 1000

bytes_por_paquete: un entero entre 50 y 1500

protocolo: valores posibles 1 (TCP), 2 (UDP), o 3 (HTTP)

seguro: valor 0 (peligroso) o 1 (seguro), generado aleatoriamente

Guarda los datos en un DataFrame de pandas.

Divide los datos en entrenamiento y prueba (80% - 20%) usando train_test_split.

self.X = self.df.drop(columns=["seguro"])
self.y = self.df["seguro"]


🏋️‍♂️ 2. Método train(self)

Entrena un modelo de KNeighborsClassifier de sklearn con los datos de entrenamiento (self.X_train, self.y_train).



📈 3. Método evaluate(self)

Evalúa el modelo sobre los datos de prueba (self.X_test, self.y_test).

Calcula y retorna la precisión del modelo (valor entre 0 y 1) usando accuracy_score.



🔍 4. Método predict(self, nuevo_dispositivo)

Recibe una lista con 3 valores: [paquetes_por_segundo, bytes_por_paquete, protocolo].

Retorna la predicción del modelo: 1 si el dispositivo es seguro, o 0 si es peligroso.



Ejemplo de uso

clasificador = IoTKNNClassifier(n_neighbors=3, n_samples=50)
clasificador.train()
print("Precisión del modelo:", clasificador.evaluate())
 
# Nuevo dispositivo IoT
nuevo = [300, 1000, 1]  # 300 paquetes/segundo, 1000 bytes/paquete, protocolo TCP
resultado = clasificador.predict(nuevo)
 
if resultado == 1:
    print("✅ Dispositivo seguro")
else:
    print("⚠️ Dispositivo peligroso")


Salida esperada

Precisión del modelo: 0.6
⚠️ Dispositivo peligroso