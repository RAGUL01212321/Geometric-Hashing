import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KDTree
from collections import defaultdict

def rmsd(V, W):
    """Compute RMSD between two sets of points."""
    return np.sqrt(np.mean(np.sum((V - W) ** 2, axis=1)))

class ProteinHashing:
    def __init__(self, bin_size=1.0, svd_components=3):
        self.hash_table = defaultdict(list)
        self.bin_size = bin_size
        self.svd = TruncatedSVD(n_components=svd_components)
        self.kd_tree = None
        self.protein_vectors = []
        self.protein_ids = []

    def apply_svd(self, features):
        return self.svd.fit_transform(features)

    def hash_function(self, reduced_features):
        return tuple(np.floor(reduced_features / self.bin_size).astype(int))

    def insert_protein(self, protein_id, coordinates):
        # Ensure coordinates are in 2D shape (n_points, 3)
        coordinates = np.array(coordinates, dtype=np.float32).reshape(-1, 3)

        # Apply SVD on the entire coordinate set at once
        reduced_features = self.apply_svd(coordinates)

        for feature in reduced_features:
            key = self.hash_function(feature)
            self.hash_table[key].append((protein_id, feature))

        # Store vectors in correct format
        self.protein_vectors.extend(reduced_features.tolist())
        self.protein_ids.extend([protein_id] * len(reduced_features))

    def build_kd_tree(self):
        if self.protein_vectors:
            vectors = np.array(self.protein_vectors, dtype=np.float32)
            print("KD-Tree Input Shape:", vectors.shape)
            self.kd_tree = KDTree(vectors)

    def query_protein(self, query_coordinates, top_k=5):
        query_reduced = self.apply_svd(np.array(query_coordinates).reshape(-1, 3))
        candidates = []
        for feature in query_reduced:
            key = self.hash_function(feature)
            if key in self.hash_table:
                candidates.extend(self.hash_table[key])

        # Use KD-Tree for nearest neighbor refinement
        if self.kd_tree:
            _, indices = self.kd_tree.query(query_reduced, k=top_k)
            refined_matches = [(self.protein_ids[i], self.protein_vectors[i]) for i in indices[0]]
            return refined_matches
        return candidates

# Load CSV Data
csv_path = "C:\\Amrita_S2\\DSA proj\\Idea2.0\\Normalized_Protein_Coordinates\\1A3N_ca_coordinates_normalized.csv"
protein_data = pd.read_csv(csv_path)

# Initialize and Insert Data
gh = ProteinHashing(bin_size=1.0, svd_components=3)
gh.insert_protein('Protein_1A3N', protein_data.values)  # Insert the new data
gh.build_kd_tree()

# Sample Query Point
query_points = np.array([[0.3, 0.9, 0.4]])
matches = gh.query_protein(query_points)
print("Matches found:", matches)
