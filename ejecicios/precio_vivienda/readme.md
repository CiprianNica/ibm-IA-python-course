Predicción del Precio de una Vivienda
🎯 Objetivo

Desarrollar un sistema de predicción que estime el precio de una vivienda en función de sus características (como superficie, número de habitaciones, antigüedad, etc.) utilizando regresión lineal. El sistema debe construirse completamente con clases en Python.



🧱 Paso 1: Simular los datos

Clase: SimuladorViviendas

Esta clase se encargará de generar un conjunto de datos sintéticos con pandas y numpy.

✔️ Instrucciones:

La clase debe llamarse SimuladorViviendas.

Debe tener un método generar_datos() que devuelva un DataFrame con 200 registros y las siguientes columnas:




🔍 Paso 2: Crear el modelo de predicción

Clase: ModeloPrecioVivienda

Esta clase representará el modelo de regresión lineal. Se encargará del entrenamiento, evaluación y predicción.

✔️ Métodos obligatorios:

entrenar(data: pd.DataFrame):

Separa los datos en variables independientes y la variable objetivo (Precio).

Divide los datos en entrenamiento (80%) y prueba (20%).

Entrena un modelo de regresión lineal con scikit-learn.

evaluar():

Muestra el error cuadrático medio (MSE).

Muestra el coeficiente de determinación R².

predecir(nueva_vivienda: pd.DataFrame) -> float:

Recibe un DataFrame con las características de una vivienda.

Devuelve el precio estimado.



🧪 Paso 3: Probar todo en conjunto

Clase: TestModeloPrecio

Esta clase servirá como lanzador general para probar que todo funcione correctamente. Dentro del método ejecutar() debe:

Generar los datos usando SimuladorViviendas.

Entrenar y evaluar el modelo con ModeloPrecioVivienda.

Crear una vivienda de ejemplo (por ejemplo: superficie 120 m², 3 habitaciones, 10 años de antigüedad, 5 km al centro, 2 baños).

Imprimir el precio estimado.



✅ Requisitos técnicos

Usar pandas, numpy y scikit-learn (LinearRegression, train_test_split, mean_squared_error, r2_score).

Las clases deben estar bien documentadas.

El código debe poder ejecutarse de principio a fin sin errores.



🧪 Ejemplo de uso

test = TestModeloPrecio()
test.ejecutar()


Salida esperada

Primeras filas de datos simulados:
    Superficie  Habitaciones  Antigüedad  Distancia_centro  Baños  \
0   87.454012             4          32          9.810270      3   
1  145.071431             3          39          7.713692      1   
2  123.199394             1           9         12.089466      2   
3  109.865848             4          42          2.476958      2   
4   65.601864             4          43         19.513501      2   
 
          Precio  
0  259267.477436  
1  314958.241175  
2  467942.246565  
3  238538.397746  
4  238781.280758  
Modelo entrenado correctamente.
 
Error Cuadrático Medio (MSE): 14748907009.71
R² del modelo: 0.02
 
El precio estimado de la vivienda es: $284,716.76