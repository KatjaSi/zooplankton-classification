import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import ipdb
import cv2

def load_features(output_file):
    features = []
    with open(output_file, 'rb') as f:
        while True:
            try:
                features.append(np.load(f, allow_pickle=True))
            except EOFError:
                break
    return np.concatenate(features, axis=0)

def load_filenames(filename_file):
    with open(filename_file, 'r') as f:
        filenames = f.read().splitlines()
    return filenames


features = load_features('features2.npy')
filenames = load_filenames('features2_filenames2.txt')
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


kmeans = KMeans(n_clusters=50)  # Adjust the number of clusters as needed
clusters_ = kmeans.fit_predict(features_scaled)

clusters = {}
for i in range(50):
    cluster_members = np.where(clusters_ == i)[0]
    clusters[i] = [filenames[member] for member in cluster_members]
    print(f'Cluster {i} has {len(cluster_members)} images.')

def visualize_and_save_images_from_clusters(clusters, folder_path, output_base_dir):
    # Create the base directory where all clusters will be saved
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for cluster_idx, image_paths in clusters.items():
        # Create a directory for each cluster
        cluster_dir = os.path.join(output_base_dir, f'cluster_{cluster_idx}')
        os.makedirs(cluster_dir, exist_ok=True)

        for img_path in image_paths:
            # Load the image
            img = cv2.imread(os.path.join(folder_path, img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Save the image to the cluster directory
            output_file_path = os.path.join(cluster_dir, os.path.basename(img_path))
            cv2.imwrite(output_file_path, img)

        print(f'Cluster {cluster_idx} saved with {len(image_paths)} images.')

# Example usage
folder_path = 'datasets/ZooScan77_small/train'  # Your input folder with images
output_base_dir = 'output_clusters'  # Base directory to save the clusters
visualize_and_save_images_from_clusters(clusters, folder_path, output_base_dir)

