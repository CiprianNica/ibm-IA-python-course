Recomendador de lenguajes de programación
🧠 Proyecto: ¿Qué lenguaje de programación debo usar?

Imagina que trabajas como consultor de software para una aceleradora de startups.

Tu tarea es desarrollar un sistema inteligente que, en base a ciertas características de un nuevo proyecto tecnológico, recomiende el lenguaje de programación más adecuado.

Para ello, cuentas con un histórico de proyectos clasificados según el lenguaje usado: Python, JavaScript, Java o C++.

Cada proyecto tiene las siguientes características numéricas:

velocidad: qué tan rápido debe ser el desarrollo (0.0 a 1.0)

mantenimiento: importancia del mantenimiento a largo plazo (0.0 a 1.0)

libs: disponibilidad de librerías relevantes para el proyecto (0.0 a 1.0)

tipo_app: tipo de aplicación:

0 = Ciencia de Datos

1 = Aplicación Web

2 = Sistema Embebido

rendimiento: necesidad de alto rendimiento (0.0 a 1.0)

Tu tarea consiste en:

✅ Objetivo

1.- Clase LanguagePredictor

Implementa una clase llamada LanguagePredictor que actúe como un sistema de recomendación de lenguajes de programación. Esta clase debe cumplir con los siguientes requisitos:

En su método __init__, debe:

Crear una instancia del modelo RandomForestClassifier de sklearn.ensemble.

Definir un diccionario label_map que asocie los valores numéricos utilizados como etiquetas con los nombres de los lenguajes de programación:

{
    0: "Python",
    1: "JavaScript",
    2: "Java",
    3: "C++"
}
Debe incluir un método .train(X, y) que:

Reciba dos arreglos de NumPy (X con características y y con etiquetas).

Entrene el modelo de Random Forest con esos datos.

Debe incluir un método .predict(features) que:

Reciba un vector de características (np.ndarray) correspondiente a un nuevo proyecto.

Devuelva el nombre del lenguaje recomendado como una cadena, usando el mapeo definido en label_map.

Esta clase permitirá entrenar un modelo de aprendizaje automático con datos sintéticos y realizar predicciones comprensibles sobre qué lenguaje usar en futuros proyectos tecnológicos.



2.- Función generate_dataset(n_samples=100, seed=42)

Implementa una función llamada generate_dataset que genere un conjunto de datos sintético representando distintos proyectos tecnológicos. Esta función debe:

Recibir dos parámetros:

n_samples (entero): número de muestras o proyectos a generar. Por defecto es 100.

seed (entero): semilla para controlar la aleatoriedad y asegurar la reproducibilidad. Por defecto es 42.

Generar, para cada proyecto, un vector de 5 características numéricas aleatorias:

velocidad: qué tan rápido debe desarrollarse el proyecto (valor entre 0.0 y 1.0)

mantenimiento: importancia del mantenimiento a largo plazo (valor entre 0.0 y 1.0)

libs: disponibilidad de librerías relevantes (valor entre 0.0 y 1.0)

tipo_app: tipo de aplicación, representado como un entero aleatorio en el rango [0, 2]:

0: Ciencia de Datos

1: Aplicación Web

2: Sistema Embebido

rendimiento: necesidad de alto rendimiento (valor entre 0.0 y 1.0)

Asignar a cada proyecto una etiqueta numérica correspondiente al lenguaje más adecuado según las siguientes reglas lógicas:

if rendimiento > 0.8 and tipo_app == 2:
    lenguaje = 3  # C++
elif mantenimiento > 0.7 and tipo_app == 1:
    lenguaje = 2  # Java
elif libs > 0.6 and tipo_app == 0:
    lenguaje = 0  # Python
else:
    lenguaje = 1  # JavaScript
Retornar dos objetos numpy.ndarray:

X: matriz con las características de todos los proyectos generados (tamaño n_samples x 5).

y: vector con las etiquetas numéricas (0 a 3) asociadas a cada proyecto, donde:

0: Python

1: JavaScript

2: Java

3: C++

Esta función sirve como generador de datos de entrenamiento para el modelo de predicción de lenguajes.



🧪 Ejemplo de uso

# Generar datos y entrenar
X, y = generate_dataset()
predictor = LanguagePredictor()
predictor.train(X, y)
 
# Crear un proyecto nuevo
new_project = np.array([0.7, 0.9, 0.5, 1, 0.6])  # Características del proyecto
 
# Predecir lenguaje ideal
pred = predictor.predict(new_project)
print(f"Lenguaje recomendado para el nuevo proyecto: {pred}")


Salida esperada

Lenguaje recomendado para el nuevo proyecto: JavaScript
