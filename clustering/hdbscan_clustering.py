import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import hdbscan
import ipdb

def load_features(output_file='features.npy'):
    features = []
    with open(output_file, 'rb') as f:
        while True:
            try:
                features.append(np.load(f, allow_pickle=True))
            except EOFError:
                break
    return np.concatenate(features, axis=0)

features = load_features('features.npy')
#filenames = load_filenames('features_filenames.txt')
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

pca = PCA(n_components=100)
features_reduced = pca.fit_transform(features_scaled)


clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10)
clusters = clusterer.fit_predict(features_reduced)
print(len(clusters))
