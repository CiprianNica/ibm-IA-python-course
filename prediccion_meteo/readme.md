Predicción meteorológica con gráfico
Los meteorólogos recopilan datos de humedad y presión atmosférica para predecir si lloverá o no.

Tu misión es construir un sistema inteligente que pueda predecir la probabilidad de lluvia utilizando un modelo de regresión logística.

🧩 Pasos a seguir

1. Crea la clase WeatherRecord

Esta clase representará un registro meteorológico individual, con:

humidity: humedad relativa en porcentaje (float)

pressure: presión atmosférica en hPa (float)

will_rain: 1 si lloverá, 0 si no (int)

Agrega un método to_vector(self) que devuelva una lista con [humidity, pressure].



2. Crea la clase WeatherDataGenerator

Esta clase generará datos sintéticos con la siguiente lógica:

Alta humedad y baja presión aumentan la probabilidad de lluvia.

Baja humedad y alta presión indican baja probabilidad de lluvia.

Implementa:

Un constructor que reciba el número de muestras (num_samples).

Un método generate() que devuelva una lista de WeatherRecord.

Puedes usar la fórmula siguiente para estimar la probabilidad de lluvia y aplicar la función sigmoide

rain_prob = (humidity - 50) * 0.03 - (pressure - 1010) * 0.02
rain_prob = 1 / (1 + np.exp(-rain_prob))
Convierte estas probabilidades en 0 o 1 usando un umbral de 0.5.



3. Crea la clase WeatherRainClassifier

Esta clase entrenará un modelo de regresión logística con scikit-learn.

Implementa:

fit(records): entrena el modelo a partir de una lista de WeatherRecord.

predict(humidity, pressure): devuelve 0 o 1 según la predicción.

evaluate(records): imprime matriz de confusión y reporte de clasificación.



4. Crea la clase WeatherRainPredictionExample

Esta clase representará el flujo completo de uso.

Implementa un método run() que:

Genere datos usando WeatherDataGenerator.

Divida los datos en entrenamiento y prueba.

Entrene un modelo con WeatherRainClassifier.

Evalúe el modelo con los datos de prueba.

Haga una predicción para condiciones nuevas (ej. humedad = 80, presión = 995).

Muestre un gráfico (matplotlib) de los datos de prueba, coloreando según si llovió o no.



🎯 Ejemplo de uso

example = WeatherRainPredictionExample()
example.run()


📊 Salida esperada

[[21  0]
 [ 0 39]]
              precision    recall  f1-score   support
 
           0       1.00      1.00      1.00        21
           1       1.00      1.00      1.00        39
 
    accuracy                           1.00        60
   macro avg       1.00      1.00      1.00        60
weighted avg       1.00      1.00      1.00        60
 
🔍 Predicción para condiciones nuevas:
   Humedad: 80%
   Presión: 995 hPa
   ¿Lloverá?: Sí ☔

