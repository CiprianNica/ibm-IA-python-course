Detectar correo electrónico spam
Clasificación de Emails: ¿Spam o No Spam?

Contexto: Tienes un conjunto de datos que contiene información sobre emails. Cada email tiene un conjunto de características, como la longitud del mensaje, la frecuencia de ciertas palabras clave, la cantidad de enlaces, y otros aspectos relevantes. El objetivo es construir un modelo de clasificación para predecir si un email es Spam o No Spam.

Objetivo: Tu tarea es implementar un modelo de clasificación que, dada la información de un email (características como la longitud del mensaje y la frecuencia de palabras clave), sea capaz de predecir si el email es Spam (1) o No Spam (0).

Funciones a Implementar:

Generar datos de emails:

Función: generar_datos_emails(num_muestras)

Esta función debe generar un conjunto de datos ficticios con num_muestras emails.

Cada email tendrá las siguientes características:

longitud_mensaje: Un número aleatorio que representa la longitud del email en caracteres (entre 50 y 500).

frecuencia_palabra_clave: Un número aleatorio que representa la frecuencia de una palabra clave relacionada con spam (entre 0 y 1).

cantidad_enlaces: Un número aleatorio que representa la cantidad de enlaces en el email (entre 0 y 10).

Cada email será etiquetado como Spam (1) o No Spam (0).

Entrenar el modelo SVM:

Función: entrenar_modelo_svm(datos, etiquetas)

Esta función debe tomar un conjunto de datos con características de emails y sus etiquetas, y entrenar un modelo de clasificación.

La salida debe ser el modelo entrenado.

Realizar predicciones:

Función: predecir_email(modelo, longitud_mensaje, frecuencia_palabra_clave, cantidad_enlaces)

Esta función debe tomar un modelo entrenado y las características de un nuevo email, y devolver si el email es Spam o No Spam.

La salida debe ser una cadena de texto que indique si el email es Spam o No Spam.

Instrucciones:

Generar Datos: Para empezar, debes generar un conjunto de datos con emails etiquetados (Spam o No Spam).

Entrenar el Modelo: Entrenar el modelo de clasificación basado en las características del email.

Predicciones: Utiliza el modelo entrenado para predecir si un email es Spam o No Spam según sus características.