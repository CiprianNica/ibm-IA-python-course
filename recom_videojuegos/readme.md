Recomendador de videojuegos
🧠 Descripción:

Vas a construir un sistema de recomendación de videojuegos que pueda predecir si a un jugador le gustará o no un videojuego basándose en características como la acción, la estrategia, los gráficos o la dificultad.

Para ello, utilizarás:

Datos sintéticos generados con numpy

Un modelo de clasificación usando Random Forest de sklearn



🕹️ Objetivo:

Crear una clase VideoGame que represente un videojuego con características numéricas.

Generar una lista de videojuegos con etiquetas (le gusta/no le gusta) usando reglas sencillas.

Entrenar un modelo con RandomForestClassifier.

Usar el modelo para predecir si un nuevo videojuego será del gusto de un jugador.



🧩 Especificaciones del ejercicio:

Crea una clase VideoGame con los siguientes atributos:

action (nivel de acción, float de 0 a 1)

strategy (nivel de estrategia, float de 0 a 1)

graphics (calidad gráfica, float de 0 a 1)

difficulty (nivel de dificultad, float de 0 a 1)

liked (opcional: 1 si le gusta al jugador, 0 si no)

Crea una clase VideoGameGenerator que se se encargará de generar videojuegos aleatorios con sus características. Para ello:

Usa la función np.random.uniform(0, 1) para generar cada característica (action, strategy, graphics, difficulty).

Redondea los valores a dos decimales con np.round(..., 2).

Calcula automáticamente si al jugador le gustará el juego, usando una regla simple:

liked = int((action > 0.6 or graphics > 0.7) and difficulty < 0.7)

Cada objeto generado debe ser una instancia de la clase VideoGame, incluyendo su etiqueta liked.

Crea la clase VideoGameClassifier :

Entrene un modelo de clasificación usando RandomForestClassifier de scikit-learn, con el parámetro n_estimators=100, que indica que se usarán 100 árboles en el bosque aleatorio.

Pueda predecir si le gustará un nuevo videojuego al jugador, a partir de sus características numéricas (action, strategy, graphics, difficulty).

Crea una clase de ejemplo VideoGameRecommendationExample  donde:

Generas 100 videojuegos aleatorios con VideoGameGenerator.

Entrena un modelo con esos datos.

Crea un nuevo videojuego con las siguientes características:

new_game = VideoGame(action=0.9, strategy=0.4, graphics=0.8, difficulty=0.3)

Predice si le gustará al jugador.

Muestra por pantalla las características del juego y si se predice que gustará o no.



✅ Ejemplo de uso

example = VideoGameRecommendationExample()
example.run()
Salida esperada

🎮 Nuevo juego:
Action: 0.9, Strategy: 0.4, Graphics: 0.8, Difficulty: 0.3
✅ Le gustará al jugador el juego? Si!
