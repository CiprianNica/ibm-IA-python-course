Clasificar piezas industriales
🎯 Objetivo general

Desarrollar un sistema automático de inspección de calidad para piezas industriales usando aprendizaje automático. Para ello, implementarás varias clases en Python que simulan la creación de datos, entrenan un modelo de clasificación (SVM) y visualizan los resultados.

Este proyecto se validará con tests automáticos, por lo tanto, las clases y métodos deben tener exactamente los nombres y firmas indicadas.



📦 Clases y métodos obligatorios

1. class Piece

Representa una pieza individual con sus características físicas y su etiqueta de calidad.

Constructor:

def __init__(self, texture, symmetry, edges, center_offset, label):

texture (float): Nivel de textura/homogeneidad (entre 0 y 1).

symmetry (float): Grado de simetría (entre 0 y 1).

edges (float): Número de bordes detectados.

center_offset (float): Desviación del centro respecto al ideal.

label (str): "Correcta" o "Defectuosa".

Método requerido:

def to_vector(self) -> list:
    # Devuelve [texture, symmetry, edges, center_offset]




2. class PieceDatasetGenerator

Genera una lista de objetos Piece simulando datos industriales con una lógica de clasificación basada en reglas.

Constructor:

def __init__(self, n=400):

n: número de piezas a generar (por defecto 400).

Método requerido:

def generate(self) -> list:
    # Devuelve una lista de objetos Piece, cada uno con su etiqueta calculada
💡 Lógica de generación:

Cada característica se genera aleatoriamente según distribuciones normales:


🔎 Reglas de clasificación:

Una pieza será etiquetada como "Defectuosa" si cumple al menos una de estas condiciones:

symmetry < 0.4 y center_offset > 0.25,

o bien texture < 0.35,

o bien edges < 30,

o bien center_offset > 0.35.

En caso contrario, será etiquetada como "Correcta"



3. class PieceClassifier

Entrena un modelo de clasificación usando SVM y permite evaluar y predecir etiquetas de nuevas piezas.

Constructor:

def __init__(self):

Métodos requeridos:

def fit(self, pieces: list) -> None:
    # Entrena el modelo SVM con una lista de objetos Piece
def predict(self, texture, symmetry, edges, center_offset) -> str:
    # Predice si una pieza con esas características es "Correcta" o "Defectuosa"
def evaluate(self, test_data: list) -> None:
    # Muestra matriz de confusión e informe de clasificación (usa sklearn)
El modelo debe usar:

from sklearn.svm import SVC
SVC(kernel='rbf', gamma='scale', C=1.0)




4. class PieceAnalysisExample

Clase demostrativa que conecta todas las partes del proyecto y muestra un ejemplo completo de uso del sistema.

Método requerido:

def run(self) -> None:

Este método debe realizar todo el flujo de trabajo del sistema:

✅ Flujo completo requerido:

Generación de datos:

Crear un objeto PieceDatasetGenerator (usar valor por defecto: 400 piezas).

Llamar a .generate() para obtener las piezas.

División de datos:

Usar train_test_split de sklearn.model_selection.

Separar en 70% entrenamiento y 30% test.

Usar random_state=42.

Entrenamiento:

Crear un PieceClassifier.

Llamar a .fit() con los datos de entrenamiento.

Evaluación:

Llamar a .evaluate() con los datos de prueba.

Mostrar matriz de confusión e informe de clasificación.

Predicción personalizada:

Predecir la clase de una pieza con estas características:

(0.45, 0.5, 45, 0.15)

Mostrar por pantalla las características y el resultado predicho.

Visualización:

Crear un DataFrame con los siguientes campos:

"Textura", "Simetría", "Bordes", "Offset", "Etiqueta"

Crear un scatter plot:

Eje X: "Textura"

Eje Y: "Offset"

Colores: verde = "Correcta", rojo = "Defectuosa"

Agregar título: "🏭 Clasificación de piezas industriales"

Mostrar leyenda y rejilla





🎯 Ejemplo de uso

example = PieceAnalysisExample()
example.run()


Salida esperada

📊 Matriz de confusión:
[[87  0]
 [28  5]]
 
📝 Informe de clasificación:
              precision    recall  f1-score   support
 
    Correcta       0.76      1.00      0.86        87
  Defectuosa       1.00      0.15      0.26        33
 
    accuracy                           0.77       120
   macro avg       0.88      0.58      0.56       120
weighted avg       0.82      0.77      0.70       120
 
 
🔎 Predicción de pieza personalizada:
  → Textura: 0.45, Simetría: 0.50, Bordes: 45, Offset: 0.15
  → Clasificación: Correcta
