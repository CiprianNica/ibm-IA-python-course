Clasificador de calidad del aire
Contexto

Trabajas en una empresa de tecnología verde que quiere monitorizar la calidad del aire para proteger la salud de las personas. Debes crear un modelo que, a partir de medidas de contaminantes en el aire, prediga si un área es saludable o está contaminada.



Objetivo

Construir un sistema en Python que:

Genere datos sintéticos de calidad del aire con medidas de PM2.5, PM10, O3 y NO2.

Entrene un clasificador SVM para distinguir entre aire saludable (0) y contaminado (1).

Permita predecir la calidad del aire de nuevas muestras.



Requisitos técnicos

1. Clase AirSample

Representa una muestra de calidad del aire con los siguientes atributos:

pm25: concentración de partículas finas PM2.5 (µg/m³)

pm10: concentración de partículas gruesas PM10 (µg/m³)

o3: concentración de ozono (ppb)

no2: concentración de dióxido de nitrógeno (ppb)

quality: etiqueta binaria (0 = saludable, 1 = contaminado). Solo se usa en datos de entrenamiento.

Método obligatorio:

to_vector(): retorna una lista o array con las cuatro medidas [pm25, pm10, o3, no2].



2. Clase AirDataGenerator

Genera datos sintéticos para entrenamiento.

Constructor: __init__(self, num_samples=200) → define cuántas muestras generar.

Método: generate(self) → retorna una lista de objetos AirSample.

Regla para asignar calidad:

if pm25 > 35 or pm10 > 50 or no2 > 40:
    quality = 1  # contaminado
else:
    quality = 0  # saludable
Notas importantes:

Para reproducibilidad, fija la semilla de NumPy con np.random.seed(42) dentro del método generate.

Usa np.random.uniform para generar valores aleatorios dentro de los rangos:

pm25: 5 a 100

pm10: 10 a 150

o3: 10 a 100

no2: 5 a 80



3. Clase AirQualityClassifier

Entrena y usa un modelo SVM para clasificar muestras.

Constructor: __init__(self) → crea un modelo SVM (sklearn.svm.SVC) con parámetros por defecto.

Método: fit(self, samples) → recibe una lista de AirSample con calidad definida, y entrena el modelo.

Método: predict(self, sample) → recibe un objeto AirSample sin etiqueta y devuelve la predicción (0 o 1).



4. Clase AirQualityExample

Ejemplo completo de uso.

Método: run(self) que:

Crea un generador AirDataGenerator con 200 muestras.

Genera datos de entrenamiento.

Entrena el clasificador AirQualityClassifier con los datos generados.

Crea una nueva muestra con valores fijos (ejemplo: pm25=22, pm10=30, o3=50, no2=35).

Predice y muestra por pantalla la calidad del aire con un mensaje claro.





Ejemplo de uso

example = AirQualityExample()
example.run()
Salida esperada

🌍 Muestra de aire:
PM2.5: 22, PM10: 30, O3: 50, NO2: 35
✅ Predicción de calidad: Saludable ✅
