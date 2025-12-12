Detectar jugadores con potencial profesional
🎮 Ejercicio: ¿Quién será un jugador profesional? - Clasificación con SVM

🧠 Contexto:

Imagina que trabajas en una plataforma de eSports y tu equipo está desarrollando un sistema de scouting para detectar jugadores con potencial profesional en base a sus estadísticas de juego.

Tu tarea es construir un modelo de clasificación usando SVM (Support Vector Machine) que, dada la información de un jugador, prediga si tiene el perfil de jugador profesional (1) o casual (0).

📋 Objetivo del ejercicio

Implementar un clasificador que:

Use datos simulados de jugadores (partidas ganadas, horas jugadas, precisión, velocidad de reacción, estrategia).

Entrene un modelo de SVM con scikit-learn.

Clasifique a nuevos jugadores como “profesional” o “casual”.

Evalúe el rendimiento del modelo utilizando la métrica de precisión (accuracy_score).

📁 Datos de entrada:

Cada jugador se representa con las siguientes características (todas normalizadas entre 0 y 1):



🧪 Ejemplo de datos

simulator = GameSimulator()
simulator.run()
Salida esperada

Jugador profesional:
Precisión del modelo: 1.00


🛠️ Tareas a realizar:

Implementa la clase ProPlayerClassifier con los métodos:

train(X, y) para entrenar el modelo.

predict(player_stats) para predecir si un jugador es profesional.

evaluate(X_test, y_test): evalúa el modelo con precisión.

Usa sklearn.svm.SVC como modelo base.

Prueba el modelo con al menos dos predicciones de distintos jugadores.

Evalúa su rendimiento con accuracy_score de sklearn.metrics






Se han guardado todos los cambios

