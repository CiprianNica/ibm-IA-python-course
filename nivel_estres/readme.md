Predicción del nivel de estrés
🌍 Contexto

Los niveles de estrés afectan directamente a la salud física y mental.

En este proyecto, trabajarás como si fueras parte del equipo de desarrollo de un sistema de monitoreo de estrés para deportistas de alto rendimiento o trabajadores en ambientes exigentes.

Se te ha encomendado diseñar un clasificador que, a partir de tres medidas fisiológicas, pueda predecir el nivel de estrés de una persona.

Para ello, deberás simular datos realistas, entrenar un modelo de aprendizaje automático y visualizar los resultados.



🎯 Objetivos

Simular datos fisiológicos (ritmo cardíaco, nivel de cortisol y conductancia de la piel).

Clasificar el nivel de estrés de las personas como:
🟢 Bajo, 🟠 Moderado o 🔴 Alto.

Entrenar un clasificador Random Forest.

Evaluar el rendimiento del modelo.

Realizar predicciones personalizadas.

Visualizar los datos y resultados con gráficos interpretables.



🛠️ Requisitos Técnicos

Debes usar:

NumPy para generar datos.

Pandas para manipular estructuras.

matplotlib.pyplot para visualizar.

sklearn para entrenamiento del modelo y métricas.

Programación orientada a objetos (clases bien definidas).



👨‍🔬 Parte 1: Clase para representar individuos

Crea una clase llamada Individual con los siguientes atributos:

Ritmo cardíaco (heart_rate) en pulsaciones por minuto.

Nivel de cortisol (cortisol_level) en µg/dL.

Conductancia de la piel (skin_conductance) en µS.

Nivel de estrés (stress_level): cadena de texto ('Bajo', 'Moderado' o 'Alto').

Incluye un método to_vector() que devuelva solo las tres primeras variables como lista.



🧪 Parte 2: Simulador de datos

Crea una clase StressDataGenerator que genere una lista de objetos Individual con valores aleatorios realistas:

Ritmo cardíaco: media 75, desviación estándar 15.

Cortisol: media 12, desviación estándar 4.

Conductancia: media 5, desviación estándar 1.5.

Clasifica los individuos según estas reglas:

🔴 Alto: si cualquiera de las tres medidas supera estos umbrales:

Ritmo cardíaco > 90

Cortisol > 18

Conductancia > 6.5

🟠 Moderado: si alguna supera:

Ritmo cardíaco > 70

Cortisol > 10

Conductancia > 4.5
pero no cumple los criterios de "Alto".

🟢 Bajo: si ninguna medida supera esos valores.



🤖 Parte 3: Clasificador con Random Forest

Crea una clase StressClassifier con los métodos:

fit(individuals) → entrena el modelo con datos.

predict(heart_rate, cortisol, conductance) → devuelve el nivel de estrés estimado.

evaluate(test_data) → imprime matriz de confusión e informe de clasificación.



🔍 Parte 4: Ejecución completa del análisis

Crea una clase llamada StressAnalysisExample que se encargue de ejecutar todo el flujo del proyecto. Esta clase debe implementar un método run() que realice las siguientes tareas:

Generación de datos:
Genera 300 individuos simulados usando la clase StressDataGenerator.

Entrenamiento y evaluación del modelo:
Divide los datos en dos subconjuntos: 70% para entrenamiento y 30% para prueba.
Entrena un clasificador usando la clase StressClassifier.
Evalúa el rendimiento del modelo mostrando:

La matriz de confusión.

El informe de clasificación con precisión, recall y f1-score.

Predicción personalizada:
Utiliza el modelo entrenado para predecir el nivel de estrés de un individuo con las siguientes características:

Ritmo cardíaco: 95

Cortisol: 20

Conductancia: 7
Muestra por pantalla la predicción realizada.

Visualización de los datos:
Convierte los datos generados en un DataFrame de pandas.
Crea un gráfico de dispersión con matplotlib:

Eje X: nivel de cortisol.

Eje Y: ritmo cardíaco.

Color de los puntos según el nivel de estrés:

🟢 Verde → Bajo

🟠 Naranja → Moderado

🔴 Rojo → Alto
Agrega título, leyenda y cuadrícula para facilitar la interpretación visual.





✅ Ejemplo de uso

example = StressAnalysisExample()
example.run()


Salida esperada

📊 Matriz de confusión:
[[33  0  0]
 [ 0  2  1]
 [ 0  0 54]]
 
📝 Informe de clasificación:
              precision    recall  f1-score   support
 
        Alto       1.00      1.00      1.00        33
        Bajo       1.00      0.67      0.80         3
    Moderado       0.98      1.00      0.99        54
 
    accuracy                           0.99        90
   macro avg       0.99      0.89      0.93        90
weighted avg       0.99      0.99      0.99        90
 
 
🧠 Predicción para individuo personalizado:
  Ritmo cardíaco: 95, Cortisol: 20, Conductancia: 7
  → Nivel estimado de estrés: Alto
