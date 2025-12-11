from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class App:
    def __init__(self, app_name, monthly_users, avg_session_length, retention_rate, social_shares, success=None):
        self.app_name = app_name
        self.monthly_users = monthly_users
        self.avg_session_length = avg_session_length
        self.retention_rate = retention_rate
        self.social_shares = social_shares
        self.success = success  # 1 o 0 si es conocida

    def to_features(self):
        return [self.monthly_users, self.avg_session_length, self.retention_rate, self.social_shares]


class AppDataset:
    def __init__(self, apps):
        self.apps = apps

    def get_feature_matrix(self):
        return [app.to_features() for app in self.apps]

    def get_target_vector(self):
        return [app.success for app in self.apps if app.success is not None]


class SuccessPredictor:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def train(self, dataset: AppDataset):
        X = dataset.get_feature_matrix()
        y = dataset.get_target_vector()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, app: App):
        features = [app.to_features()]
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)[0]
        
    def predict_proba(self, app: App):
        features = [app.to_features()]
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[0][1]


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
new_app = App("StudyBoost", 4000, 15.0, 0.5, 700)
predicted_success = predictor.predict(new_app)
prob = predictor.predict_proba(new_app)

print(f"¿Será exitosa la app {new_app.app_name}? {'Sí' if predicted_success else 'No'}")
print(f"Probabilidad estimada de éxito: {prob:.2f}")


