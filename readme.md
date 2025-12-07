Predicción del desgaste de vehículos
🛠️ Misión: Predicción del Desgaste de Vehículos Militares

Como analista de datos en una base militar, tu tarea es predecir el nivel de desgaste de vehículos en función de las horas de uso.

Esta herramienta será clave para evitar fallos operativos y optimizar los mantenimientos preventivos.



🎯 Objetivo

Implementa un sistema basado en regresión lineal que:

Genere registros de entrenamiento con la clase VehicleDataGenerator.

Represente cada registro con la clase VehicleRecord, que almacene:

hours_used: horas de servicio del vehículo.

wear_level: nivel de desgaste en porcentaje (0 a 100).

Entrene un modelo de regresión lineal con la clase VehicleWearRegressor, que:

Reciba una lista de registros.

Aprenda la relación entre horas de uso y desgaste.

Permita hacer predicciones para nuevas horas.

Visualice los datos y prediga el desgaste con la clase VehicleWearPredictionExample.



🧱 Organización en clases



🚗 VehicleRecord

Representa un registro individual del uso de un vehículo.

Atributos:

hours_used: horas de servicio del vehículo.

wear_level: nivel de desgaste en porcentaje (0 a 100).

Método:

to_vector() → Devuelve una lista o vector con [hours_used] para usarlo en el modelo.

🧩 Pista:

Esta clase funciona como una “fila” de datos, que luego será usada para entrenar el modelo.



🧮 VehicleDataGenerator

Genera un conjunto de datos sintéticos (simulados) para entrenamiento.

Atributo:

num_samples: cantidad de registros a generar.

Método:

generate() → Devuelve una lista de objetos VehicleRecord.

El desgaste debe aumentar con las horas de uso, pero con cierta variabilidad aleatoria.
Puedes usar:

np.random.uniform() → para generar horas aleatorias entre 50 y 500.

np.random.normal() → para añadir ruido (variación natural).

np.clip() → para limitar el desgaste entre 0 y 100.

🧩 Pista:

Una posible relación lineal podría ser algo así:

desgaste = 10 + 0.18 * horas + ruido_aleatorio



🧰 VehicleWearRegressor

Entrena un modelo de regresión lineal que relacione horas de uso y desgaste.

Métodos:

fit(records): recibe una lista de VehicleRecord y entrena el modelo.
Convierte los datos en arrays X y y usando numpy.

predict(hours): devuelve el desgaste estimado para un número dado de horas.

get_model(): devuelve el modelo entrenado (LinearRegression).



📊 VehicleWearPredictionExample

Clase principal para ejecutar la simulación completa.

Método:

run(), que debe:

Generar los datos con VehicleDataGenerator.

Entrenar el modelo con VehicleWearRegressor.

Predecir el desgaste para un vehículo con 250 horas de uso.

Mostrar los resultados por pantalla.

Visualizar la relación con un gráfico:

Puntos verdes → Datos reales.

Línea roja → Línea de regresión.

Línea gris vertical → Nuevo vehículo.

🧩 Pista visual:

Usa matplotlib.pyplot para graficar los puntos (plt.scatter) y la línea (plt.plot).



⚙️ Tecnologías a usar

NumPy, Pandas, Matplotlib

LinearRegression de scikit-learn

