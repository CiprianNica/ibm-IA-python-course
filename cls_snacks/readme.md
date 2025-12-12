Clasificador de snacks saludables
Objetivo:

En este ejercicio, aprenderás a crear un clasificador para predecir si un snack es saludable o no, basándote en características nutricionales como las calorías, azúcar, proteínas, grasas y fibra.

Usaremos un árbol de decisión para crear un modelo que prediga si un snack es saludable en función de estos atributos.



Descripción:

Imagina que trabajas en una aplicación de salud que recomienda snacks a los usuarios. Tienes acceso a un conjunto de datos que contiene información sobre varios snacks y su contenido nutricional.

Usaremos estos datos para entrenar un modelo que pueda predecir si un snack es saludable basándose en sus atributos.



Pasos a seguir:

Creación de la clase Snack:

Define una clase Snack que tenga los siguientes atributos: calories, sugar, protein, fat, fiber, y un atributo opcional is_healthy, que será el resultado que queremos predecir (1 si el snack es saludable, 0 si no lo es).

Crea un método to_vector() que convierta un snack en un vector de características (calorías, azúcar, proteínas, grasas, fibra).

Generación de Datos Sintéticos con la clase SnackGenerator:

Crea una clase SnackGenerator que sea capaz de generar un conjunto de datos sintéticos con snacks. Esta clase debe crear entre 50 y 200 snacks con valores aleatorios para las características mencionadas.

Para que los datos sean realistas, utiliza valores dentro de los siguientes rangos:

Calorías: entre 50 y 500.

Azúcar: entre 0 y 50 gramos.

Proteína: entre 0 y 30 gramos.

Grasa: entre 0 y 30 gramos.

Fibra: entre 0 y 15 gramos.

La variable is_healthy debe seguir una regla aproximada: un snack es saludable si tiene menos de 200 calorías, menos de 15 gramos de azúcar, menos de 10 gramos de grasa, y al menos 5 gramos de proteína o fibra.

Clasificador de Snacks con Árbol de Decisión:

Crea una clase SnackClassifier que use un árbol de decisión para clasificar los snacks.

Esta clase debe tener dos métodos:

fit(): entrenar el modelo usando un conjunto de snacks y sus etiquetas (is_healthy).

predict(): predecir si un snack específico es saludable o no.

Crear un Ejemplo de Uso:

Crea un objeto de la clase SnackRecommendationExample que entrene el clasificador utilizando el generador de snacks.

Luego, crea un snack de prueba con valores nutricionales conocidos, como 150 calorías, 10 gramos de azúcar, 6 gramos de proteína, 5 gramos de grasa y 3 gramos de fibra.

Usa el clasificador para predecir si este snack es saludable y muestra la predicción.



🔁 Nota: La clase SnackRecommendationExample debe contener todo el flujo de uso del sistema: generación de datos, entrenamiento del clasificador, predicción de un nuevo snack e impresión del resultado.

⚠️ Consejo: Asegúrate de que todos los atributos usados para entrenar y predecir estén en el mismo orden y formato (números, no strings).



Requisitos:

Uso de Árbol de Decisión: Para realizar la clasificación, usa la librería sklearn y su DecisionTreeClassifier.

Generación de datos: Usa numpy para generar valores aleatorios.

Impresión de resultados: Imprime la información nutricional del snack de prueba junto con la predicción de si es saludable o no.



Resultado esperado:

Al ejecutar el código, el sistema debe mostrar la información nutricional del snack de prueba y una predicción indicando si es saludable o no.

Ejemplo de uso

# Ejecutar ejemplo
example = SnackRecommendationExample()
example.run()
Salida esperada

🔍 Snack Info:
Calories: 150, Sugar: 10g, Protein: 6g, Fat: 5g, Fiber: 3g
✅ Predicción: Este snack no es saludable.
