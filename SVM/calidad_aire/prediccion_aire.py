import numpy as np
from sklearn.svm import SVC

class AirSample:
    def __init__(self, pm25, pm10, o3, no2, quality=None):
        self.pm25 = pm25
        self.pm10 = pm10
        self.o3 = o3
        self.no2 = no2
        self.quality = quality
    def to_vector(self):
        return [self.pm25, self.pm10, self.o3, self.no2]

class AirDataGenerator:
    def __init__(self, num_samples=200):
        self.num_samples = num_samples
    def generate(self):
        np.random.seed(42)
        samples = []
        for _ in range(self.num_samples):
            pm25 = np.random.uniform(5, 100)
            pm10 = np.random.uniform(10, 150)
            o3 = np.random.uniform(10, 100)
            no2 = np.random.uniform(5, 80)
            if pm25 > 35 or pm10 > 50 or no2 > 40:
                quality = 1
            else:
                quality = 0
            samples.append(AirSample(pm25, pm10, o3, no2, quality))
        return samples

class AirQualityClassifier:
    def __init__(self):
        self.model = SVC()
    def fit(self, samples):
        X = [s.to_vector() for s in samples]
        y = [s.quality for s in samples]
        self.model.fit(X, y)
    def predict(self, sample):
        return self.model.predict([sample.to_vector()])[0]

class AirQualityExample:
    def run(self):
        generator = AirDataGenerator(200)
        samples = generator.generate()
        clf = AirQualityClassifier()
        clf.fit(samples)
        new_sample = AirSample(22, 30, 50, 35)
        pred = clf.predict(new_sample)
        print("🌍 Muestra de aire:")
        print(f"PM2.5: {new_sample.pm25:.0f}, PM10: {new_sample.pm10:.0f}, O3: {new_sample.o3:.0f}, NO2: {new_sample.no2:.0f}")
        if pred == 0:
            print("✅ Predicción de calidad: Saludable ✅")
        else:
            print("❌ Predicción de calidad: Contaminado ❌")

if __name__ == "__main__":
    example = AirQualityExample()
    example.run()
