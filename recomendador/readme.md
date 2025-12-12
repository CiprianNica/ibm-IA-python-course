Recomendador de Personajes
🎮 "Recomendador de Personajes: ¿Qué tipo de personaje deberías elegir?"

📘 Enunciado

En este ejercicio trabajarás como desarrollador de sistemas inteligentes para un nuevo videojuego tipo RPG online. El juego permite a los jugadores crear personajes y elegir entre distintos roles o clases (por ejemplo: guerrero, mago, arquero, curandero…).

Tu tarea es construir un modelo de recomendación que, dado un perfil de jugador (nivel, estilo de combate, número de partidas jugadas, etc.), recomiende qué tipo de personaje debería usar, basándose en datos históricos de otros jugadores similares.

🧩 Requerimientos

Crea una clase Player que represente a un jugador con los siguientes atributos:

name: nombre del jugador.

level: nivel del jugador (1 a 100).

aggressiveness: valor entre 0 y 1 que representa su estilo ofensivo.

cooperation: valor entre 0 y 1 que representa cuánto coopera con el equipo.

exploration: valor entre 0 y 1 que representa cuánto le gusta explorar el mapa.

preferred_class: clase de personaje que suele elegir (solo en los datos de entrenamiento).

Implementa un método .to_features() en la clase para convertir al jugador en una lista de características numéricas (sin la clase preferida).

Crea una clase PlayerDataset que contenga una lista de jugadores y proporcione:

get_X() → lista de listas de características.

get_y() → lista de clases preferidas.

Crea una clase ClassRecommender que use KNN para:

Entrenar el modelo a partir de un PlayerDataset.

Predecir la mejor clase para un nuevo jugador (predict(player)).

Obtener los k jugadores más parecidos (get_nearest_neighbors(player)).

(Opcional) Permite probar diferentes valores de k y evaluar la precisión del modelo con cross_val_score.



🧪 Ejemplo de uso

# Datos de entrenamiento
players = [
    Player("Alice", 20, 0.8, 0.2, 0.1, "Warrior"),
    Player("Bob", 45, 0.4, 0.8, 0.2, "Healer"),
    Player("Cleo", 33, 0.6, 0.4, 0.6, "Archer"),
    Player("Dan", 60, 0.3, 0.9, 0.3, "Healer"),
    Player("Eli", 50, 0.7, 0.2, 0.9, "Mage"),
    Player("Fay", 25, 0.9, 0.1, 0.2, "Warrior"),
]
 
# Nuevo jugador
new_player = Player("TestPlayer", 40, 0.6, 0.3, 0.8)
 
# Entrenamiento y predicción
dataset = PlayerDataset(players)
recommender = ClassRecommender(n_neighbors=3)
recommender.train(dataset)
 
# Resultado
recommended_class = recommender.predict(new_player)
neighbors_indices = recommender.get_nearest_neighbors(new_player)
 
print(f"Clase recomendada para {new_player.name}: {recommended_class}")
print("Jugadores similares:")
for i in neighbors_indices:
    print(f"- {players[i].name} ({players[i].preferred_class})")


🧪 Salida esperada

Clase recomendada para TestPlayer: Archer
Jugadores similares:
- Bob (Healer)
- Cleo (Archer)
- Eli (Mage)