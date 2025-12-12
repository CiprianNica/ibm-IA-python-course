Recomendador de canciones inteligente
🧠 Contexto

Estás desarrollando un sistema para una plataforma musical que quiere ofrecer recomendaciones automáticas basadas en características cuantitativas de las canciones, como su energía o duración.

Utilizarás el algoritmo K-Nearest Neighbors (KNN) de la biblioteca scikit-learn para encontrar las canciones más similares a una canción objetivo.



🎯 Objetivo del ejercicio

Implementar un sistema de recomendación de canciones en Python, usando el modelo de K Vecinos Más Cercanos de scikit-learn.

El sistema debe permitir recomendar canciones similares a partir de características musicales numéricas.



📌 Requisitos

🧩 1. Clase Song

Crea una clase Song que represente una canción, con los siguientes atributos:

title (str): título de la canción.

artist (str): artista o grupo musical.

energy (float): energía de la canción (0.4 a 1.0).

danceability (float): cuán bailable es la canción (0.4 a 1.0).

duration (int): duración en segundos (180 a 300).

popularity (int): nivel de popularidad (50 a 100).

La clase debe incluir:

Un método to_vector() que devuelva una lista con los valores [energy, danceability, duration, popularity].

Un método __str__() que permita imprimir la canción en formato "Song Title by Artist".



🤖 2. Clase SongRecommender

Crea una clase SongRecommender que use el algoritmo de KNN de scikit-learn:

El constructor debe aceptar un parámetro k (número de vecinos a considerar).

El método fit(song_list) debe:

Convertir la lista de canciones en una matriz de características numéricas.

Ajustar el modelo NearestNeighbors con estas características.

El método recommend(target_song) debe:

Obtener los k vecinos más cercanos a la canción objetivo.

Devolver la lista de canciones recomendadas (sin incluir la propia canción objetivo si aparece).



🔁 3. Clase SongGenerator

Crea una clase SongGenerator con:

Un parámetro num_songs (por defecto 30).

Un método generate() que genere canciones aleatorias con numpy, usando nombres como "Song1", "Song2", etc., y artistas "Artist1", "Artist2", etc.



🧪 4. Clase SongRecommendationExample

Crea una clase de ejemplo que:

Genere una lista de canciones con SongGenerator.

Defina una canción personalizada como objetivo (target_song).

Cree una instancia de SongRecommender, la entrene con las canciones y obtenga recomendaciones.

Imprima por pantalla las canciones recomendadas.



Ejemplo de salida:

example = SongRecommendationExample()
example.run()
Salida esperada

🎵 Recomendaciones para 'Mi Canción':
 - Song29 by Artist4
 - Song11 by Artist1
 - Song25 by Artist5


💡 Recomendaciones para completar el ejercicio

Usa numpy para generar valores aleatorios.

Recuerda importar NearestNeighbors desde sklearn.neighbors.

Asegúrate de convertir los objetos Song a vectores antes de ajustar o predecir con el modelo.

No incluyas la canción objetivo entre las recomendaciones (verifica si es necesario).

