Agrupar viajeros según sus preferencias
🧠 Contexto

Imagina que trabajas en una agencia de viajes internacional que recibe cientos de perfiles de clientes.

Cada viajero indica cuánto le gustan distintos tipos de destinos:

🏖️ Playa

🏔️ Montaña

🏙️ Ciudad

🌄 Campo

Tu misión es desarrollar un sistema que agrupe automáticamente a los viajeros en tres grandes tipos según sus gustos.

Para lograrlo, utilizarás el algoritmo de K-Means Clustering de scikit-learn.



🎯 Objetivo del ejercicio

Debes implementar cuatro clases principales para estructurar tu solución:

1.  Traveler (almacena las preferencias de un viajero)

Atributos:

beach (int): preferencia por la playa (0–10)

mountain (int): preferencia por la montaña (0–10)

city (int): preferencia por la ciudad (0–10)

countryside (int): preferencia por el campo (0–10)

Método:

to_vector(self) -> list: devuelve las preferencias del viajero como una lista [beach, mountain, city, countryside].



2. TravelerGenerator (genera viajeros aleatorios)

Atributos:

num_travelers (int): cantidad de viajeros a generar.

Método:

generate(self) -> list[Traveler]: genera una lista de objetos Traveler con preferencias aleatorias.
Para cada preferencia, usa:

np.random.randint(0, 11)  # genera valores enteros entre 0 y 10 (inclusive)



3. TravelerClusterer (agrupa a los viajeros con K-Means)

Atributos:

model: instancia de KMeans con n_clusters=3 y random_state=42.

Métodos:

fit(self, travelers: list[Traveler]): entrena el modelo de KMeans con los vectores de preferencias.

predict(self, traveler: Traveler) -> int: devuelve el número de clúster (0, 1 o 2) al que pertenece un nuevo viajero.

get_cluster_centers(self) -> np.ndarray: retorna los centros de los clústeres calculados por el modelo.



4. TravelerClusteringExample (orquesta todo el flujo del ejemplo)

Método:

run(self): debe hacer lo siguiente:

Generar 200 viajeros usando TravelerGenerator.

Entrenar un modelo TravelerClusterer.

Mostrar en pantalla los centros de los 3 clústeres, indicando el promedio de preferencias en cada uno.

Crear un nuevo viajero personalizado, por ejemplo:

new_traveler = Traveler(beach=9, mountain=2, city=8, countryside=1)

Predecir a qué clúster pertenece ese viajero con predict.

Mostrar en pantalla los resultados.



✅ Ejemplo de uso

# Ejecutar ejemplo
example = TravelerClusteringExample()
example.run()


Salida esperada

🏝️🏔️🏙️🌄 Cluster Centers (Preferencias promedio):
Cluster 0: Playa=4.79, Montaña=5.16, Ciudad=7.79, Campo=7.82
Cluster 1: Playa=5.11, Montaña=5.54, Ciudad=6.60, Campo=1.66
Cluster 2: Playa=4.69, Montaña=5.23, Ciudad=1.46, Campo=6.16
 
Interpretación aproximada:
- Cluster con alta Playa y Ciudad: Viajero urbano y costero.
- Cluster con alta Montaña y Campo: Amante de la naturaleza.
- Cluster equilibrado: Viajero versátil o aventurero.
 
🔍 Nuevo viajero con preferencias:
Playa: 9, Montaña: 2, Ciudad: 8, Campo: 1
📌 El nuevo viajero pertenece al grupo 1.