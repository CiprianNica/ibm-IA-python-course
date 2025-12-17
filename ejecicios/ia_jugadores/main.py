# Phantom Arena: Entrenando una IA para jugadores
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import pandas as pd

class Player:
	def __init__(self, player_name, character_type, avg_session_time, matches_played, aggressive_actions, defensive_actions, items_bought, victories, style=None):
		self.player_name = player_name
		self.character_type = character_type
		self.avg_session_time = avg_session_time
		self.matches_played = matches_played
		self.aggressive_actions = aggressive_actions
		self.defensive_actions = defensive_actions
		self.items_bought = items_bought
		self.victories = victories
		self.style = style

class GameModel:
	def __init__(self, players):
		self.players = players
		self.df = self._players_to_df(players)
		self.classification_model = None
		self.regression_model = None
		self.cluster_model = None
		self.label_encoders = {}

	def _players_to_df(self, players):
		data = [{
			'player_name': p.player_name,
			'character_type': p.character_type,
			'avg_session_time': p.avg_session_time,
			'matches_played': p.matches_played,
			'aggressive_actions': p.aggressive_actions,
			'defensive_actions': p.defensive_actions,
			'items_bought': p.items_bought,
			'victories': p.victories,
			'style': p.style
		} for p in players]
		return pd.DataFrame(data)

	def _encode_features(self, df, fit=True):
		df = df.copy()
		for col in ['character_type']:
			if fit:
				le = LabelEncoder()
				df[col] = le.fit_transform(df[col])
				self.label_encoders[col] = le
			else:
				le = self.label_encoders[col]
				df[col] = le.transform(df[col])
		return df

	def train_classification_model(self):
		df = self._encode_features(self.df, fit=True)
		X = df[['character_type', 'avg_session_time', 'matches_played', 'aggressive_actions', 'defensive_actions', 'items_bought', 'victories']]
		y = df['style']
		le_style = LabelEncoder()
		y_enc = le_style.fit_transform(y)
		self.label_encoders['style'] = le_style
		self.classification_model = LogisticRegression(max_iter=200)
		self.classification_model.fit(X, y_enc)

	def train_regression_model(self):
		df = self._encode_features(self.df, fit=True)
		X = df[['character_type', 'avg_session_time', 'matches_played', 'aggressive_actions', 'defensive_actions', 'items_bought']]
		y = df['victories']
		self.regression_model = RandomForestRegressor(random_state=42)
		self.regression_model.fit(X, y)

	def train_clustering_model(self, n_clusters=2):
		df = self._encode_features(self.df, fit=True)
		X = df[['character_type', 'avg_session_time', 'matches_played', 'aggressive_actions', 'defensive_actions', 'items_bought', 'victories']]
		self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
		self.cluster_model.fit(X)
		self.df['cluster'] = self.cluster_model.labels_

	def predict_style(self, player):
		df = self._players_to_df([player])
		df = self._encode_features(df, fit=False)
		X = df[['character_type', 'avg_session_time', 'matches_played', 'aggressive_actions', 'defensive_actions', 'items_bought', 'victories']]
		pred = self.classification_model.predict(X)[0]
		le_style = self.label_encoders['style']
		return le_style.inverse_transform([pred])[0]

	def predict_victories(self, player):
		df = self._players_to_df([player])
		df = self._encode_features(df, fit=False)
		X = df[['character_type', 'avg_session_time', 'matches_played', 'aggressive_actions', 'defensive_actions', 'items_bought']]
		pred = self.regression_model.predict(X)[0]
		return pred

	def assign_cluster(self, player):
		df = self._players_to_df([player])
		df = self._encode_features(df, fit=False)
		X = df[['character_type', 'avg_session_time', 'matches_played', 'aggressive_actions', 'defensive_actions', 'items_bought', 'victories']]
		return int(self.cluster_model.predict(X)[0])

	def print_players_by_cluster(self):
		if self.cluster_model is None or 'cluster' not in self.df:
			print("El modelo de clustering no ha sido entrenado.")
			return
		for cluster in sorted(self.df['cluster'].unique()):
			print(f"Cluster {cluster}:")
			grupo = self.df[self.df['cluster'] == cluster]
			for _, row in grupo.iterrows():
				print(f"{row['player_name']} - {row['character_type'].capitalize()} - {row['style'].capitalize() if pd.notnull(row['style']) else 'N/A'}")

# Ejemplo de uso
if __name__ == "__main__":
	# Crear datos de prueba para varios jugadores
	players_data = [
		Player("P1", "mage", 40, 30, 90, 50, 20, 18, "aggressive"),
		Player("P2", "tank", 60, 45, 50, 120, 25, 24, "strategic"),
		Player("P3", "archer", 50, 35, 95, 60, 22, 20, "aggressive"),
		Player("P4", "tank", 55, 40, 60, 100, 28, 22, "strategic"),
	]
	model = GameModel(players_data)
	model.train_classification_model()
	model.train_regression_model()
	model.train_clustering_model(n_clusters=2)
	# Crear un nuevo jugador para realizar predicciones
	new_player = Player("TestPlayer", "mage", 42, 33, 88, 45, 21, 0)
	predicted_style = model.predict_style(new_player)
	predicted_victories = model.predict_victories(new_player)
	predicted_cluster = model.assign_cluster(new_player)
	print(f"Estilo de juego predicho para {new_player.player_name}: {predicted_style}")
	print(f"Victorias predichas para {new_player.player_name}: {predicted_victories:.2f}")
	print(f"Cluster asignado a {new_player.player_name}: {predicted_cluster}")
	# Mostrar jugadores por cluster
	print("\nJugadores por cluster:")
	model.print_players_by_cluster()
