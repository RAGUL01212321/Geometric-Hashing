import numpy as np
import pandas as pd
from scipy.spatial import distance
import hashlib

# Load reduced coordinates
file_path = r"C:\Amrita_S2\DSA proj\Idea2.0\Reduced_Coordinates\1A3N_ca_coordinates_reduced.csv"
data = pd.read_csv(file_path).values

# Feature Extraction Function
def extract_features(coords):
    features = []
    n = len(coords)

    # 1. Pairwise Distances
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance.euclidean(coords[i], coords[j])
            features.append(dist)

    # 2. Centroid Distances
    centroid = np.mean(coords, axis=0)
    for point in coords:
        centroid_dist = distance.euclidean(point, centroid)
        features.append(centroid_dist)

    # 3. Angle Features (Optional)
    def compute_angle(a, b, c):
        ab = np.array(b) - np.array(a)
        bc = np.array(c) - np.array(b)
        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        return np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    for i in range(1, n - 1):
        angle = compute_angle(coords[i - 1], coords[i], coords[i + 1])
        features.append(angle)

    return np.array(features)

# Hash Key Generation Function
def generate_hash_key(features):
    feature_str = "_".join(map(str, features))
    return hashlib.sha256(feature_str.encode()).hexdigest()[:10]  # 10-char key

# Build Hash Table
def build_hash_table(protein_data):
    hash_table = {}
    for protein_id, coords in protein_data.items():
        features = extract_features(coords)
        hash_key = generate_hash_key(features)
        if hash_key not in hash_table:
            hash_table[hash_key] = []
        hash_table[hash_key].append(protein_id)
    return hash_table

# Example Usage
protein_data = {"1A3N": data}  # Sample dataset format
hash_table = build_hash_table(protein_data)

print("Hash Table:", hash_table)
