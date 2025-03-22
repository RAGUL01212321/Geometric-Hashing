import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from joblib import Parallel, delayed
import hashlib
import os
from sklearn.preprocessing import StandardScaler

# Path to protein coordinate dataset
dataset_path = r"C:\Amrita_S2\DSA proj\Idea2.0\Protein_Coordinates_V2.0"

# Load, Normalize, and Apply SVD
def load_and_preprocess_data(path, n_components=2):
    protein_data = {}
    svd = TruncatedSVD(n_components=n_components)
    
    for file in os.listdir(path):
        if file.endswith(".csv"):
            file_path = os.path.join(path, file)
            try:
                data = pd.read_csv(file_path).values
                if data.shape[0] == 0 or data.shape[1] != 3:
                    continue  # Skip empty or invalid files
                normalized_data = StandardScaler().fit_transform(data)
                reduced_data = svd.fit_transform(normalized_data)
                protein_id = file.split(".")[0]  # Extract protein ID
                protein_data[protein_id] = reduced_data
            except Exception as e:
                print(f"Error reading file {file}: {e}")
                continue
    return protein_data

# Feature Extraction using Vectorization
def extract_features(coords):
    pairwise_distances = distance_matrix(coords, coords)
    upper_triangle = np.triu(pairwise_distances, k=1)  # Extract upper triangle (no duplicates)
    distances = upper_triangle[upper_triangle > 0]      # Flatten non-zero distances
    centroid = np.mean(coords, axis=0)
    centroid_distances = np.linalg.norm(coords - centroid, axis=1)
    features = np.concatenate((distances, centroid_distances))
    return features

# Hash Key Generation
def generate_hash_key(features):
    feature_str = "_".join(map(str, features))
    return hashlib.sha256(feature_str.encode()).hexdigest()[:10]

# Build Hash Table with Parallel Processing
def process_protein(file, path):
    file_path = os.path.join(path, file)
    data = pd.read_csv(file_path).values
    if data.shape[0] == 0 or data.shape[1] != 3:
        return None, None

    normalized_data = StandardScaler().fit_transform(data)
    reduced_data = randomized_svd(normalized_data, n_components=2)[0]  # Efficient SVD
    features = extract_features(reduced_data)
    hash_key = generate_hash_key(features)

    protein_id = file.split(".")[0]
    return protein_id, hash_key

def build_hash_table_parallel(path):
    hash_table = {}
    results = Parallel(n_jobs=-1)(  # -1 uses all CPU cores
        delayed(process_protein)(file, path) for file in os.listdir(path) if file.endswith(".csv")
    )
    
    for protein_id, hash_key in results:
        if protein_id and hash_key:
            if hash_key not in hash_table:
                hash_table[hash_key] = []
            hash_table[hash_key].append(protein_id)
    return hash_table

# Execution
protein_data = load_and_preprocess_data(dataset_path)
hash_table = build_hash_table_parallel(dataset_path)

print("Hash Table:", hash_table)


# Search DB
# Search Function
def search_protein(query_coords, hash_table):
    # Normalize query data
    query_normalized = StandardScaler().fit_transform(query_coords)
    
    # Apply SVD to reduce dimensions
    query_reduced = randomized_svd(query_normalized, n_components=2)[0]
    
    # Extract features
    query_features = extract_features(query_reduced)
    
    # Generate hash key for the query
    query_hash_key = generate_hash_key(query_features)
    
    # Search in the hash table
    if query_hash_key in hash_table:
        print(f"Protein(s) found: {hash_table[query_hash_key]}")
    else:
        print("No matching protein found.")

# Example Usage
query_coordinates = np.array([
    [10.228, 20.761, 6.807],
    [6.624, 21.451, 7.763],
    [4.831, 23.237, 4.928],
    [2.252, 25.966, 5.311],
    [-0.457, 23.367, 4.513],
    [1.069, 20.972, 7.101],
    [0.946, 23.686, 9.764],
    [-2.657, 24.552, 8.903],
    [-3.578, 20.829, 9.048],
    [-1.868, 20.308, 12.382],
    [-3.565, 23.346, 13.951],
    [-6.957, 22.358, 12.597],
    [-6.647, 18.789, 13.878],
    [-5.212, 19.711, 17.304],
    [-7.757, 22.528, 17.555],
    [-10.567, 19.934, 17.19],
    [-8.862, 17.802, 19.902],
    [-9.346, 20.831, 22.076],
    [-9.976, 20.02, 25.73],
    [-9.621, 16.236, 25.2],
    [-5.867, 16.641, 24.592],
    [-4.88, 15.262, 28.037],
    [-7.238, 12.272, 27.73],
    [-5.834, 11.496, 24.246],
    [-2.242, 11.796, 25.459],
    [-2.984, 9.188, 28.214],
    [-4.711, 6.9, 25.656],
    [-1.702, 7.161, 23.318],
    [0.667, 6.141, 26.19]])

search_protein(query_coordinates, hash_table)
