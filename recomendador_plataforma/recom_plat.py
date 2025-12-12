from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

class Project:
    def __init__(self, name, team_size, budget, duration_months, realtime_required, needs_offline, target_users, recommended_platform=None):
        self.name = name
        self.team_size = team_size
        self.budget = budget
        self.duration_months = duration_months
        self.realtime_required = realtime_required
        self.needs_offline = needs_offline
        self.target_users = target_users
        self.recommended_platform = recommended_platform

    def to_features(self, label_encoder=None):
        # Codifica target_users si se pasa un encoder
        target_users_enc = label_encoder.transform([self.target_users])[0] if label_encoder else self.target_users
        return [
            self.team_size,
            self.budget,
            self.duration_months,
            int(self.realtime_required),
            int(self.needs_offline),
            target_users_enc
        ]

class ProjectDataset:
    def __init__(self, projects):
        self.projects = projects
        self._fit_encoders()
    def _fit_encoders(self):
        self.target_users_encoder = LabelEncoder()
        self.platform_encoder = LabelEncoder()
        # Solo proyectos históricos para entrenar encoders
        historicos = [p for p in self.projects if p.recommended_platform is not None]
        self.target_users_encoder.fit([p.target_users for p in self.projects])
        self.platform_encoder.fit([p.recommended_platform for p in historicos])
    def get_X_y(self):
        historicos = [p for p in self.projects if p.recommended_platform is not None]
        X = [p.to_features(self.target_users_encoder) for p in historicos]
        y = self.platform_encoder.transform([p.recommended_platform for p in historicos])
        return X, y
    def decode_platform(self, label):
        return self.platform_encoder.inverse_transform([label])[0]

class PlatformRecommender:
    def __init__(self):
        self.clf = DecisionTreeClassifier(random_state=42)
        self.dataset = None

    def train(self, dataset: ProjectDataset):
        self.dataset = dataset
        X, y = dataset.get_X_y()
        self.clf.fit(X, y)

    def predict(self, project: Project):
        X_new = [project.to_features(self.dataset.target_users_encoder)]
        pred_label = self.clf.predict(X_new)[0]
        return self.dataset.decode_platform(pred_label)

class PlatformRecommenderVisualizer:
    def plot_tree(self, recommender: PlatformRecommender):
        plt.figure(figsize=(20,10))
        tree.plot_tree(recommender.clf, 
                    feature_names=['team_size', 'budget', 'duration_months', 'realtime_required', 'needs_offline', 'target_users'], 
                    class_names=recommender.dataset.platform_encoder.classes_,
                    filled=True)
        plt.show()  
        
        
        
        
# Ejemplo de uso
if __name__ == "__main__":
    # Proyectos históricos
    projects = [
        Project("WebApp1", 5, 50, 6, False, False, "empresa", "web"),
        Project("MobileApp1", 8, 80, 8, True, False, "global", "mobile"),
        Project("DesktopApp1", 4, 40, 10, False, True, "local", "desktop"),
        Project("WebApp2", 6, 60, 7, False, False, "empresa", "web"),
        Project("MobileApp2", 10, 100, 12, True, False, "global", "mobile"),
    ]
    # Nuevo proyecto
    new_project = Project("AIChatApp", 7, 70, 9, True, True, "empresa")
    dataset = ProjectDataset(projects)
    recommender = PlatformRecommender()
    recommender.train(dataset)
    pred_platform = recommender.predict(new_project)
    print(f"Plataforma recomendada para '{new_project.name}': {pred_platform}")
