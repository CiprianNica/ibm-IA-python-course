Diseñar una IA que entienda a los jugadores
Título: "Phantom Arena: Entrenando una IA para clasificar, predecir y agrupar jugadores"

Descripción:

En este ejercicio, deberás entrenar un modelo de Machine Learning para predecir el estilo de juego de un jugador, el número de victorias esperadas y asignar a cada jugador un grupo de características. Para ello, deberás crear una clase llamada GameModel que gestione y entrene los modelos correspondientes.

En lugar de cargar datos desde un archivo CSV, recibirás un conjunto de datos de prueba directamente en el código. Usarás estos datos para entrenar y evaluar tus modelos.

Tareas:

Crear la clase Player: Esta clase representará a un jugador y debe contener los siguientes atributos:

player_name: nombre del jugador (string).

character_type: tipo de personaje (string). Puede ser "mage", "tank", "archer", "assassin".

avg_session_time: tiempo promedio por sesión en minutos (float).

matches_played: número total de partidas jugadas (int).

aggressive_actions: cantidad de acciones agresivas realizadas (int).

defensive_actions: cantidad de acciones defensivas realizadas (int).

items_bought: cantidad de objetos comprados (int).

victories: número de victorias (int).

style: estilo de juego del jugador ("aggressive" o "strategic", sólo uno de estos).

Crear la clase GameModel: Esta clase debe ser capaz de:

Recibir una lista de jugadores y almacenarlos.

Entrenar tres modelos diferentes:

Modelo de clasificación: para predecir el estilo de juego del jugador (aggressive o strategic).

Modelo de regresión: para predecir el número de victorias de un jugador.

Modelo de clustering: para asignar a cada jugador un grupo basado en sus características.

Proveer métodos para:

Predecir el estilo de juego de un jugador.

Predecir las victorias de un jugador.

Asignar un cluster a un jugador.

Pruebas:

No deberás usar ningún archivo externo. Todos los datos serán proporcionados directamente en el código, en forma de una lista de objetos Player.

Después de entrenar los modelos, deberás hacer predicciones para un jugador de prueba.

🧩 Datos de prueba

Se te proporcionará un conjunto de datos de prueba como el siguiente:

players_data = [
    Player("P1", "mage", 40, 30, 90, 50, 20, 18, "aggressive"),
    Player("P2", "tank", 60, 45, 50, 120, 25, 24, "strategic"),
    Player("P3", "archer", 50, 35, 95, 60, 22, 20, "aggressive"),
    Player("P4", "tank", 55, 40, 60, 100, 28, 22, "strategic"),
]


🧩 Ejemplo de uso

# Crear datos de prueba para varios jugadores
players_data = [
    Player("P1", "mage", 40, 30, 90, 50, 20, 18, "aggressive"),
    Player("P2", "tank", 60, 45, 50, 120, 25, 24, "strategic"),
    Player("P3", "archer", 50, 35, 95, 60, 22, 20, "aggressive"),
    Player("P4", "tank", 55, 40, 60, 100, 28, 22, "strategic"),
]
 
# Instanciar el modelo con los datos de los jugadores
model = GameModel(players_data)
 
# Entrenar los modelos
model.train_classification_model()
model.train_regression_model()
model.train_clustering_model()
 
# Crear un nuevo jugador para realizar predicciones
new_player = Player("TestPlayer", "mage", 42, 33, 88, 45, 21, 0)
 
# Realizar predicciones
predicted_style = model.predict_style(new_player)
predicted_victories = model.predict_victories(new_player)
predicted_cluster = model.assign_cluster(new_player)
 
# Imprimir los resultados de las predicciones
print(f"Estilo de juego predicho para {new_player.player_name}: {predicted_style}")
print(f"Victorias predichas para {new_player.player_name}: {predicted_victories:.2f}")
print(f"Cluster asignado a {new_player.player_name}: {predicted_cluster}")


🧩 Salida esperada

Estilo de juego predicho para TestPlayer: aggressive
Victorias predichas para TestPlayer: 17.70
Cluster asignado a TestPlayer: 0


🧩 Tarea Opcional: Mostrar jugadores por cluster

Para este ejercicio adicional, te proponemos una función que te permitirá visualizar los jugadores agrupados por clusters. Esta función será útil para explorar y entender cómo el modelo de clustering (KMeans) ha agrupado a los jugadores en función de sus características.

Descripción de la tarea:

Debes implementar una función que imprima los jugadores asignados a cada cluster después de que el modelo KMeans haya sido entrenado. Cada cluster debe ser visualizado con el nombre del jugador, el tipo de personaje y su estilo de juego.

Especificaciones:

Utiliza el modelo KMeans entrenado en el ejercicio anterior.

La función debe recorrer los diferentes clusters y mostrar los jugadores pertenecientes a cada uno, con los siguientes detalles:

Nombre del jugador

Tipo de personaje

Estilo de juego

La salida debe ser algo como:

Cluster 0:
P1 - Mage - Aggressive
P3 - Archer - Aggressive
Cluster 1:
P2 - Tank - Strategic
P4 - Tank - Strategic
Consejos:

Puedes utilizar model.cluster_model.labels_ para obtener las asignaciones de los clusters.

Convierte los datos de los jugadores en un DataFrame para facilitar la manipulación y visualización de la información.

La función debe imprimir los jugadores por cada cluster, y para ello puedes agrupar los jugadores según el valor de model.cluster_model.labels_.