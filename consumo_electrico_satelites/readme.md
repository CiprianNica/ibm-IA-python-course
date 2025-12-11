Predecir consumo energético de satélites
Estás trabajando en un proyecto de análisis orbital. Tu objetivo es crear un modelo capaz de predecir el consumo energético diario (kWh) de un satélite en función de variables como:

duración de la misión

cantidad de paneles solares

carga útil transportada

Ya cuentas con:

una clase Satellite que representa cada satélite

un generador de datos sintéticos (SatelliteDatasetGenerator)

un procesador que crea un DataFrame con los datos (SatelliteDataProcessor)

una clase que se encarga de graficar (SatellitePlotter)

una clase principal (SatelliteAnalysisExample) que orquesta todas las fases

Sin embargo, falta completar la parte esencial: el modelo de regresión lineal que predice el consumo energético.

✔️ Parte ya implementada (no debes modificar)

Satellite

SatelliteDatasetGenerator

SatelliteDataProcessor

SatellitePlotter

SatelliteAnalysisExample (salvo donde uses tu clase)

❗ Tu tarea

Debes completar la clase EnergyConsumptionRegressor:

Implementar el entrenamiento del modelo (fit).

Calcular las predicciones.

Evaluar el modelo usando R².

Obtener los coeficientes de la regresión.

Completa únicamente las partes marcadas como # TODO.