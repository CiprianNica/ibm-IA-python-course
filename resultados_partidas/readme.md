Predecir resultados en partidas multijugador
🧠 Objetivo

En este ejercicio, aplicarás tus conocimientos de regresión logística para construir un modelo capaz de predecir si un jugador ganó o perdió una partida, a partir de sus estadísticas individuales.



📋 Descripción del problema

Tienes que construir un modelo predictivo que, a partir de las estadísticas de un jugador en una partida, determine si ganó o no.

Para ello, deberás:

Crear datos sintéticos que representen partidas ficticias de jugadores.

Entrenar un modelo de regresión logística con esos datos.

Implementar una función que prediga el resultado (ganar o no) para un nuevo jugador.



📦 Paso 1: Definir una clase para representar una partida

Crea una clase PlayerMatchData con los siguientes atributos:

kills: número de enemigos eliminados

deaths: número de veces que el jugador ha muerto

assists: asistencias realizadas

damage_dealt: daño total infligido

damage_received: daño total recibido

healing_done: curación realizada

objective_time: tiempo (en segundos) que el jugador estuvo capturando objetivos

won: 1 si el jugador ganó la partida, 0 si perdió

Incluye un método .to_dict() que devuelva los datos como un diccionario (sin la variable won, opcionalmente).



📦 Paso 2: Generar datos sintéticos con NumPy

Crea una función llamada generate_synthetic_data que genere un conjunto de datos de entrenamiento simulando partidas de videojuegos. Para ello:

Utiliza la librería numpy para generar los valores numéricos.

Cada instancia representará el desempeño de un jugador en una partida.

La función debe devolver una lista de objetos PlayerMatchData (ya definida previamente).

Implementa la siguiente lógica para cada jugador:

Reglas para los datos:

kills: número de enemigos eliminados, generado con una distribución de Poisson con media 5.

kills = np.random.poisson(5)
deaths: número de veces que el jugador ha muerto, distribución de Poisson con media 3.

assists: asistencias realizadas, distribución de Poisson con media 2.

damage_dealt: daño infligido, calculado como kills * 300 + ruido aleatorio normal.

damage_received = deaths * 400 + np.random.normal(0, 100)
damage_received: daño recibido, como deaths * 400 + ruido aleatorio normal.

healing_done: cantidad de curación, valor aleatorio entero entre 0 y 300.

objective_time: tiempo (en segundos) controlando objetivos, valor aleatorio entre 0 y 120.

won: el jugador se considera que ganó la partida si hizo más daño del que recibió y tuvo más kills que muertes.

🧠 Tu función debe seguir esta estructura:

import numpy as np
 
def generate_synthetic_data(n=100):
    data = []
    for _ in range(n):
        # Genera cada variable siguiendo las instrucciones dadas
        # Crea un objeto PlayerMatchData con estos valores
        # Añádelo a la lista de datos
 
    return data


🧪 Paso 3: Crear y entrenar el modelo

Crea una clase VictoryPredictor que entrene un modelo de regresión logística con los datos sintéticos. Esta clase debe tener:

Un método train(data) para entrenar el modelo.

Un método predict(player: PlayerMatchData) que devuelva 1 si predice victoria, 0 si derrota.



📌 Ejemplo de uso

# Crear datos de entrenamiento
training_data = generate_synthetic_data(150)
 
# Entrenar modelo
predictor = VictoryPredictor()
predictor.train(training_data)
 
# Crear jugador de prueba
test_player = PlayerMatchData(8, 2, 3, 2400, 800, 120, 90, None)
 
# Predecir si ganará
prediction = predictor.predict(test_player)
print(f"¿El jugador ganará? {'Sí' if prediction == 1 else 'No'}")


Salida esperada

¿El jugador ganará? Sí