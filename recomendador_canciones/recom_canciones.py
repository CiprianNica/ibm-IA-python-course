from sklearn.neighbors import NearestNeighbors
import numpy as np

class Song:
    def __init__(self, title, artist, energy, danceability, duration, popularity):
        self.title = title
        self.artist = artist
        self.energy = energy
        self.danceability = danceability
        self.duration = duration
        self.popularity = popularity

    def to_vector(self):
        return [self.energy, self.danceability, self.duration, self.popularity]

    def __str__(self):
        return f"{self.title} by {self.artist}"



class SongGenerator:
    def __init__(self, num_songs=30):
        self.num_songs = num_songs

    def generate(self):
        np.random.seed(42)
        songs = []
        for i in range(1, self.num_songs + 1):
            title = f"Song{i}"
            artist = f"Artist{np.random.randint(1, 6)}"
            energy = np.round(np.random.uniform(0.4, 1.0), 2)
            danceability = np.round(np.random.uniform(0.4, 1.0), 2)
            duration = np.random.randint(180, 301)
            popularity = np.random.randint(50, 101)
            songs.append(Song(title, artist, energy, danceability, duration, popularity))
        return songs



class SongRecommender:
    def __init__(self, k=3):
        self.k = k
        self.model = NearestNeighbors(n_neighbors=k+1)  # +1 to exclude the song itself
        self.songs = []

    def fit(self, song_list):
        self.songs = song_list
        X = [song.to_vector() for song in song_list]
        self.model.fit(X)

    def recommend(self, target_song):
        target_vec = [target_song.to_vector()]
        distances, indices = self.model.kneighbors(target_vec)
        rec_indices = [i for i in indices[0] if self.songs[i] != target_song][:self.k]
        return [self.songs[i] for i in rec_indices]



class SongRecommendationExample:
    def run(self):
        generator = SongGenerator(num_songs=30)
        songs = generator.generate()
        target_song = Song("Mi Canción", "Mi Artista", 0.75, 0.8, 215, 88)
        recommender = SongRecommender(k=3)
        recommender.fit(songs + [target_song])
        recommendations = recommender.recommend(target_song)
        print(f"🎵 Recomendaciones para '{target_song.title}':")
        for song in recommendations:
            print(f"- {song}")



if __name__ == "__main__":
    example = SongRecommendationExample()
    example.run()