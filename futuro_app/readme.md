Predecir futuro de una app
¿Tendrá éxito tu app?

📱 Contexto

Eres parte de un equipo de análisis de una startup que lanza apps móviles. Se te ha asignado la tarea de construir un modelo que pueda predecir si una app será exitosa o no en función de sus métricas iniciales.

La empresa ha recopilado datos de otras apps anteriores, tanto exitosas como fallidas, y quiere automatizar este análisis con Machine Learning.

🎯 Objetivo

Crea un sistema en Python que permita:

Representar los datos de una app.

Preparar un conjunto de datos a partir de múltiples apps.

Entrenar un modelo de regresión logística con scikit-learn.

Predecir si una app será exitosa.

De forma opcional, mostrar la probabilidad de éxito.

🧱 Estructura del proyecto

Debes implementar las siguientes clases:



📦 App

Representa una app móvil con las siguientes características:

app_name: nombre de la app.

monthly_users: número de usuarios mensuales.

avg_session_length: duración media de las sesiones (en minutos).

retention_rate: tasa de retención entre 0 y 1.

social_shares: número de veces que se ha compartido en redes sociales.

success: valor opcional (1 = éxito, 0 = fracaso).

Método:

to_features(self): devuelve una lista en el siguiente orden:



[monthly_users, avg_session_length, retention_rate, social_shares]



📊 AppDataset

Representa un conjunto de datos de apps.

Debe construirse a partir de una lista de objetos App.

Métodos:

get_feature_matrix(self): devuelve una matriz con las características numéricas de todas las apps.

get_target_vector(self): devuelve una lista con las etiquetas success (solo incluye aquellas que no sean None).



🤖 SuccessPredictor

Encargado de entrenar y usar el modelo de regresión logística para predecir el éxito de nuevas apps.

Métodos:

train(dataset: AppDataset): entrena el modelo usando un conjunto de datos.

predict(app: App): recibe una app y devuelve 1 si se predice éxito, 0 en caso contrario.

predict_proba(app: App): (opcional) recibe una app y devuelve la probabilidad de éxito como un número decimal entre 0 y 1.

⚠️ Nota: debes usar StandardScaler de sklearn.preprocessing para escalar los datos tanto al entrenar como al predecir. Esto mejora el rendimiento del modelo.



🧪 Ejemplo de uso

# Datos de entrenamiento
apps = [
    App("FastChat", 10000, 12.5, 0.65, 1500, 1),
    App("FitTrack", 500, 5.0, 0.2, 50, 0),
    App("GameHub", 15000, 25.0, 0.75, 3000, 1),
    App("BudgetBuddy", 800, 6.5, 0.3, 80, 0),
    App("EduFlash", 12000, 18.0, 0.7, 2200, 1),
    App("NoteKeeper", 600, 4.0, 0.15, 30, 0)
]
 
dataset = AppDataset(apps)
predictor = SuccessPredictor()
predictor.train(dataset)
 
# Nueva app a evaluar
new_app = App("StudyBoost", 20000, 15.0, 0.5, 700)
predicted_success = predictor.predict(new_app)
prob = predictor.predict_proba(new_app)
 
print(f"¿Será exitosa la app {new_app.app_name}? {'Sí' if predicted_success else 'No'}")
print(f"Probabilidad estimada de éxito: {prob:.2f}")


🧪 Salida esperada

¿Será exitosa la app StudyBoost? Sí
Probabilidad estimada de éxito: 0.83


