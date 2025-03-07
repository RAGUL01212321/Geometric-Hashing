# Search using coordinates

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

# ---------------------- STEP 1: Load All Protein Files ---------------------- #

directory_path = r"C:\Amrita_S2\DSA proj\Idea2.0\Protein_Coordinates_V1.0"
MAX_FEATURE_SIZE = 1000  # Fixed feature size

protein_data = {}
for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        protein_id = filename.split(".")[0]  
        file_path = os.path.join(directory_path, filename)

        df = pd.read_csv(file_path, usecols=['X', 'Y', 'Z'])
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)  

        if df.empty:
            print(f"⚠️ Warning: {protein_id} has no valid numeric coordinates. Skipping.")
            continue  

        coordinates = df.values
        protein_data[protein_id] = coordinates

print(f"✅ Loaded {len(protein_data)} valid protein structures.")

# ---------------------- STEP 2: Compute Geometric Features ---------------------- #

def compute_features(coords):
    """Computes geometric features with fixed vector size."""
    if coords.shape[0] < 4:  
        print(f"⚠️ Skipping a protein with only {coords.shape[0]} C-alpha atoms (requires ≥ 4).")
        return None

    pairwise_distances = squareform(pdist(coords, metric='euclidean')).flatten()

    # Compute bond angles
    def bond_angles(coords):
        vectors = np.diff(coords, axis=0)
        norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        unit_vectors = vectors / norms
        dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        return np.degrees(angles)

    # Compute dihedral angles
    def dihedral_angles(coords):
        p = coords
        b0 = -1.0 * (p[:-3] - p[1:-2])
        b1 = p[1:-2] - p[2:-1]
        b2 = p[2:-1] - p[3:]

        b1 /= np.linalg.norm(b1, axis=1)[:, np.newaxis]
        v = b0 - np.sum(b0 * b1, axis=1)[:, np.newaxis] * b1
        w = b2 - np.sum(b2 * b1, axis=1)[:, np.newaxis] * b1

        x = np.sum(v * w, axis=1)
        y = np.sum(np.cross(b1, v) * w, axis=1)
        
        return np.degrees(np.arctan2(y, x))

    angles = bond_angles(coords)
    dihedrals = dihedral_angles(coords)

    feature_vector = np.hstack([pairwise_distances, angles, dihedrals])

    # **FIX: Ensure fixed-length feature vector**
    if feature_vector.shape[0] < MAX_FEATURE_SIZE:
        feature_vector = np.pad(feature_vector, (0, MAX_FEATURE_SIZE - feature_vector.shape[0]), mode='constant')
    else:
        feature_vector = feature_vector[:MAX_FEATURE_SIZE]  # Truncate if too long

    return feature_vector

# Compute features for all valid proteins
feature_matrix = []
protein_ids = []
for protein_id, coords in protein_data.items():
    features = compute_features(coords)
    if features is not None:
        feature_matrix.append(features)
        protein_ids.append(protein_id)

# Convert to NumPy array
feature_matrix = np.vstack(feature_matrix)

print("✅ Feature matrix shape before SVD:", feature_matrix.shape)

# ---------------------- STEP 3: Apply SVD for Feature Compression ---------------------- #

svd = TruncatedSVD(n_components=min(100, feature_matrix.shape[1]))  
reduced_features = svd.fit_transform(feature_matrix)

print("✅ Reduced feature matrix shape:", reduced_features.shape)

# ---------------------- STEP 4: Store in Geometric Hash Table ---------------------- #

hash_table = defaultdict(list)

def hash_function(features):
    return tuple(np.round(features[:10], decimals=1))  

for i, protein_id in enumerate(protein_ids):
    hash_key = hash_function(reduced_features[i])
    hash_table[hash_key].append({"protein_id": protein_id, "features": reduced_features[i]})

# Display the Hash Table
print("\n🔹 Geometric Hash Table:")
for key, proteins in hash_table.items():
    print(f"Hash Key: {key}")
    for protein in proteins:
        print(f"  - Protein ID: {protein['protein_id']}")

# ---------------------- STEP 5: Query System Using Protein Coordinates ---------------------- #

def find_similar_proteins(query_features, hash_table, k=5):
    """Finds similar proteins based on query feature vector."""
    query_hash_key = hash_function(query_features)
    candidates = hash_table.get(query_hash_key, [])
    
    if not candidates:
        return "No similar proteins found in hash table."
    
    similarities = [
        (protein["protein_id"], np.linalg.norm(query_features - protein["features"]))
        for protein in candidates
    ]
    
    similarities.sort(key=lambda x: x[1])
    return similarities[:k]

# ---------------------- STEP 6: Searching Using Query Protein Coordinates ---------------------- #

query_file = r"C:\Amrita_S2\DSA proj\Reconstructed_Protein_Coordinates.csv"  # Change this for another protein search

if os.path.exists(query_file):
    query_df = pd.read_csv(query_file, usecols=['X', 'Y', 'Z'])
    query_df = query_df.apply(pd.to_numeric, errors='coerce')
    query_df.dropna(inplace=True)

    if query_df.empty:
        print("⚠️ Query protein has no valid numeric coordinates!")
    else:
        query_coordinates = query_df.values
        query_features = compute_features(query_coordinates)
        query_features_reduced = svd.transform(query_features.reshape(1, -1))

        # Search for similar proteins
        similar_proteins = find_similar_proteins(query_features_reduced[0], hash_table, k=5)
        print("\n✅ Similar proteins found:", similar_proteins)
else:
    print(f"❌ Query file not found: {query_file}")
