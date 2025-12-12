Agrupar perfiles de sueño
💤 Análisis de Perfiles de Sueño con K-Means

Una empresa especializada en salud del sueño desea analizar los hábitos de descanso de cientos de personas.

Se han recogido datos sobre cómo duermen los usuarios, y ahora tu misión es agrupar a los individuos en diferentes tipos de "dormidores" usando aprendizaje no supervisado.



🎯 Objetivo

Crear un sistema en Python que genere perfiles de sueño sintéticos, los agrupe en 3 categorías usando K-Means Clustering, y visualice los resultados.



🧩 Estructura que debes seguir

1. SleepProfile

Clase que representa el perfil de sueño de una persona.

Atributos:

duration: duración promedio del sueño en horas (ej. 7.5)

latency: tiempo promedio para quedarse dormido en minutos (ej. 25)

wakeups: cantidad de despertares durante la noche (ej. 2)

variability: variabilidad en la hora de irse a dormir, en minutos (ej. 35)

Métodos:

to_vector(self): Devuelve la información del perfil como una lista [duration, latency, wakeups, variability].



2. SleepDatasetGenerator

Clase que genera una lista de objetos SleepProfile con datos sintéticos y realistas.

Constructor:

__init__(self, n=300): genera n perfiles (por defecto 300).

Método:

generate(self): retorna una lista de objetos SleepProfile, donde los atributos se generan así:

duration: distribución normal con media 7 horas y desviación estándar 1.2 → np.random.normal(7, 1.2, n)

latency: distribución normal positiva (usar np.abs) con media 20 y std 10

wakeups: distribución Poisson con media 1.5 → np.random.poisson(1.5, n)

variability: distribución normal positiva con media 30 y std 15



3. SleepClusterer

Clase que entrena un modelo K-Means y agrupa los perfiles.

Constructor:

__init__(self, n_clusters=3): crea un modelo KMeans con 3 clústeres y un StandardScaler para escalar los datos antes de entrenar.

Métodos:

fit(self, profiles): recibe una lista de SleepProfile, los convierte a vectores, normaliza con StandardScaler y entrena el modelo. Retorna X_scaled y las etiquetas de clúster (labels).

get_cluster_centers(self): devuelve los centros de clúster transformados a su escala original (usa scaler.inverse_transform).



4. SleepAnalysisExample

Clase que ejecuta todo el análisis.

Método:

run(self):

Genera los perfiles usando SleepDatasetGenerator.

Agrupa los perfiles usando SleepClusterer con 3 grupos.

Crea un DataFrame para mostrar resultados.

Imprime los centroides (promedios de cada grupo).

Grafica los perfiles usando matplotlib (eje X: duración, eje Y: variabilidad).



🛠 Requisitos mínimos

Usar numpy para generar datos sintéticos.

Usar scikit-learn para KMeans y StandardScaler.

Usar matplotlib para graficar los resultados.

Organizar el código en clases separadas como se indica.

Mostrar en consola los centroides de cada grupo.

Graficar los grupos usando diferentes colores.



📈 ¿Qué deberías mostrar al final?

Los centroides de los grupos, interpretando lo que caracteriza a cada uno.

Un gráfico de dispersión donde se vea la agrupación de perfiles por:

Eje X: Duración del sueño

Eje Y: Variabilidad de hora de dormir

Comentarios sobre posibles tipos de durmientes: ¿hay un grupo de "insomnes"? ¿otro de "buenos durmientes"?



🧪 Ejemplo de uso

example = SleepAnalysisExample()
example.run()


Salida esperada

📌 Centroides de los grupos:
Grupo 0: Duración=6.30h, Latencia=19.3min, Despertares=1.2, Variabilidad=39.6min
Grupo 1: Duración=6.79h, Latencia=18.9min, Despertares=3.4, Variabilidad=26.5min
Grupo 2: Duración=7.98h, Latencia=18.6min, Despertares=1.0, Variabilidad=22.3min