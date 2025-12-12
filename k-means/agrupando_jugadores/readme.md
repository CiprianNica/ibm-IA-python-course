Agrupando jugadores
¿Qué tipo de gamer eres?"

📄 Enunciado del ejercicio

Eres parte del equipo de análisis de una plataforma de videojuegos que quiere entender mejor a sus usuarios. Se ha recopilado información sobre distintos jugadores basada en su comportamiento dentro del juego. Tu misión es agrupar a estos jugadores en diferentes tipos (clusters) según su estilo de juego, utilizando el algoritmo de K-Means.

🧠 Tareas a realizar

Crea una clase Player que contenga los siguientes atributos:

name (str): nombre del jugador

avg_session_time (float): tiempo medio de juego por sesión (en horas)

missions_completed (int): número de misiones completadas

accuracy (float): precisión de disparo (entre 0 y 1)

aggressiveness (float): valor entre 0 (pasivo) y 1 (muy agresivo)

Crea una clase PlayerClusterer con los siguientes métodos:

fit(players: List[Player], n_clusters: int): entrena un modelo K-Means con los datos de los jugadores.

predict(player: Player) -> int: devuelve el número de cluster al que pertenece un nuevo jugador.

get_cluster_centers(): devuelve los centros de los clusters.

print_cluster_summary(players: List[Player]): imprime qué jugadores hay en cada grupo.

Usa los datos proporcionados a continuación para entrenar el modelo con 3 clusters:

data = [
            ("Alice", 2.5, 100, 0.85, 0.3),
            ("Bob", 1.0, 20, 0.60, 0.7),
            ("Charlie", 3.0, 150, 0.9, 0.2),
            ("Diana", 0.8, 15, 0.55, 0.9),
            ("Eve", 2.7, 120, 0.88, 0.25),
            ("Frank", 1.1, 30, 0.62, 0.65),
            ("Grace", 0.9, 18, 0.58, 0.85),
            ("Hank", 3.2, 160, 0.91, 0.15)
        ]
4.   Crea una clase GameAnalytics que haga lo siguiente:

Cree los objetos Player con los datos anteriores.

Cree un objeto PlayerClusterer, entrene el modelo y muestre los clusters formados.

Prediga el cluster para un nuevo jugador: ("Zoe", 1.5, 45, 0.65, 0.5).



✅ Requisitos del ejercicio

Utiliza scikit-learn (KMeans) para la agrupación.

Usa programación orientada a objetos.

No uses ficheros externos. Todo debe estar en el código.

Asegúrate de imprimir resultados entendibles para los usuarios.



🧪 Ejemplo de uso

analytics = GameAnalytics()
analytics.run()


🧪 Salida esperada

Cluster 2:
  - Alice
  - Eve
Cluster 1:
  - Bob
  - Diana
  - Frank
  - Grace
Cluster 0:
  - Charlie
  - Hank
 
Jugador Zoe pertenece al cluster: 1