Agrupación de clientes según comportamientos de compra
🎯 Objetivo General:

Simular un conjunto de datos de clientes con características comerciales, aplicar técnicas de machine learning no supervisado para segmentarlos mediante agrupamiento (clustering), y visualizar los resultados para analizar distintos perfiles de cliente. Todo el proyecto debe estar organizado usando clases.



📦 Requisitos:

Utiliza las librerías:

numpy

matplotlib

sklearn.cluster.KMeans

sklearn.preprocessing.StandardScaler

Estructura tu código en tres clases:

SimuladorClientes

ModeloSegmentacionClientes

TestSegmentacionClientes



🧩 Clase SimuladorClientes

Esta clase debe simular los datos de clientes con las siguientes características:

Atributos esperados (por cliente):

Monto gastado: valor entre 100 y 10,000.

Frecuencia de compras: entre 1 y 100.

Categorías preferidas: 3 valores aleatorios entre 1 y 5 (representando número de compras por categoría).

Implementa el método:

def generar_datos(self) -> np.ndarray

Este método debe devolver un array de 200 muestras, cada una con 3 columnas:

Monto gastado

Frecuencia de compras

Total de categorías preferidas (suma de los 3 valores generados)



🧠 Clase ModeloSegmentacionClientes

Esta clase debe encargarse de entrenar el modelo y realizar predicciones.

Atributos:

n_clusters: número de grupos a formar (por defecto: 3).

scaler: instancia de StandardScaler.

modelo: instancia de KMeans.

Métodos requeridos:

entrenar(datos: np.ndarray) -> None:

Escala los datos con StandardScaler.

Ajusta el modelo KMeans.

Guarda los datos escalados como atributo para futuras visualizaciones.

predecir(cliente_nuevo: list) -> int:

Recibe un nuevo cliente (3 características).

Escala sus datos.

Devuelve el número de cluster al que pertenece.



🧪 Clase TestSegmentacionClientes

Clase para integrar y probar todo el sistema. Implementa el método:

def ejecutar(self) -> None

Este método debe:

Crear una instancia de SimuladorClientes y generar los datos.

Instanciar ModeloSegmentacionClientes con 3 clusters.

Entrenar el modelo con los datos simulados.

Mostrar los primeros 5 registros de los datos simulados.

Predecir el cluster para un nuevo cliente con los siguientes datos:

cliente_nuevo = [2000, 10, 12]

(Significa: gastó 2000, compra 10 veces, tiene 12 compras sumadas en sus categorías preferidas).

  6. Mostrar por pantalla el cluster al que pertenece este nuevo cliente.

  7. Incluye una visualización de los datos segmentados usando matplotlib.

Representa los clientes en un gráfico de dispersión donde:

El eje X es el monto gastado.

El eje Y es la frecuencia de compras.

Los puntos se colorean según el cluster al que pertenecen (usa modelo.modelo.labels_ para obtenerlos).

Añade etiquetas, título, y barra de color.



💡 Consejos para el alumno

Usa np.column_stack para combinar varias columnas en un array.

Escalar los datos es fundamental en clustering: sin esto, las variables dominantes como “monto gastado” podrían sesgar los grupos.

Usa KMeans(n_clusters=3, random_state=42) para asegurar reproducibilidad.



🧪 Ejemplo de uso

test = TestSegmentacionClientes()
test.ejecutar()


Salida esperada

Primeros 5 registros de datos simulados:
[[3.80794718e+03 2.40000000e+01 1.00000000e+01]
 [9.51207163e+03 7.50000000e+01 1.30000000e+01]
 [7.34674002e+03 7.20000000e+01 9.00000000e+00]
 [6.02671899e+03 3.60000000e+01 1.00000000e+01]
 [1.64458454e+03 3.80000000e+01 9.00000000e+00]]
Modelo entrenado con 3 clusters.
El nuevo cliente pertenece al cluster: 1