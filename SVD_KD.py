import numpy as np
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd

# Step 1: Parse PDB File & Extract Features
def parse_pdb(file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", file_path)
    
    atom_positions = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):  # C-alpha atom for backbone
                    atom = residue["CA"]
                    atom_positions.append(atom.coord)
    return np.array(atom_positions)

# Step 2: Compute Invariant Geometric Features
def compute_invariants(atom_positions):
    invariants = []
    transformed_positions = []
    for i in range(len(atom_positions) - 2):
        a, b, c = atom_positions[i], atom_positions[i + 1], atom_positions[i + 2]
        
        # Compute pairwise distances (geometric invariant)
        d1 = np.linalg.norm(a - b)
        d2 = np.linalg.norm(b - c)
        d3 = np.linalg.norm(c - a)
        
        # Compute angles
        v1 = b - a
        v2 = c - b
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians
        
        # Create feature vector
        feature = [d1, d2, d3, theta]
        invariants.append(feature)
        transformed_positions.append((d1, d2, theta))  # Store transformed positions for visualization
    return np.array(invariants), np.array(transformed_positions)

# Step 3: Apply SVD for Dimensionality Reduction
def apply_svd(features, n_components=2):
    U, S, Vt = svd(features, full_matrices=False)
    return U[:, :n_components]  # Reduce to n_components dimensions

# Step 4: Hashing Function for Efficient Retrieval
def generate_hash(feature_vector):
    feature_str = "_".join(map(str, feature_vector.flatten()))
    return hashlib.md5(feature_str.encode()).hexdigest()

# Step 5: Store & Retrieve Structures Efficiently
class ProteinDatabase:
    def __init__(self):
        self.feature_hashes = {}  # Dictionary for hashing
        self.kd_tree = None
        self.svd_kd_tree = None
        self.features = []
        self.svd_features = []

    def add_protein(self, pdb_file, protein_id):
        positions = parse_pdb(pdb_file)
        features, _ = compute_invariants(positions)
        svd_features = apply_svd(features)
        
        for feature, svd_feature in zip(features, svd_features):
            hash_key = generate_hash(feature)
            self.feature_hashes[hash_key] = protein_id
            self.features.append(feature)
            self.svd_features.append(svd_feature)
        
        self.kd_tree = KDTree(self.features)  # KD-Tree for full features
        self.svd_kd_tree = KDTree(self.svd_features)  # KD-Tree for reduced SVD features

    def retrieve_protein(self, query_feature, use_svd=False):
        if use_svd:
            if self.svd_kd_tree is None:
                return None
            query_svd = apply_svd(query_feature.reshape(1, -1))  # Ensure it's 2D
            _, idx = self.svd_kd_tree.query(query_svd[0])  # Extract first component correctly

        else:
            if self.kd_tree is None:
                return None
            query_svd = apply_svd(query_feature.reshape(1, -1))  # Ensure 2D transformation
            if query_svd.shape[1] != 2:  # Validate shape before querying KDTree
                raise ValueError(f"Expected 2D feature vector, got shape {query_svd.shape}")

            _, idx = self.svd_kd_tree.query(query_svd[0])  # Use correctly shaped query

        
        hash_key = generate_hash(self.features[idx])
        return self.feature_hashes.get(hash_key, "Not Found")

# Step 6: Visualization of Protein Structures
def visualize_protein_structure(atom_positions, transformed_positions, title_original="Original Protein Structure", title_transformed="Transformed Structure"):
    fig = plt.figure(figsize=(12, 5))
    
    # Original Structure
    ax1 = fig.add_subplot(121, projection='3d')
    x, y, z = atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2]
    ax1.scatter(x, y, z, c='b', marker='o', label="C-alpha Atoms")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")
    ax1.set_zlabel("Z-axis")
    ax1.set_title(title_original)
    ax1.legend()
    
    # Transformed Structure
    ax2 = fig.add_subplot(122, projection='3d')
    tx, ty, tz = transformed_positions[:, 0], transformed_positions[:, 1], transformed_positions[:, 2]
    ax2.scatter(tx, ty, tz, c='r', marker='^', label="Transformed Points")
    ax2.set_xlabel("Distance 1")
    ax2.set_ylabel("Distance 2")
    ax2.set_zlabel("Angle (radians)")
    ax2.set_title(title_transformed)
    ax2.legend()
    
    plt.show()

# Example Usage
db = ProteinDatabase()
db.add_protein("C:\\Amrita_S2\\DSA proj\\2c0k.pdb", "Protein_1")

query_feature, transformed_positions = compute_invariants(parse_pdb("C:\\Amrita_S2\\DSA proj\\2c0k.pdb"))
retrieved_protein_kd = db.retrieve_protein(query_feature[0], use_svd=False)
retrieved_protein_svd = db.retrieve_protein(query_feature[0], use_svd=True)
print("Retrieved Protein (KD-Tree):", retrieved_protein_kd)
print("Retrieved Protein (SVD + KD-Tree):", retrieved_protein_svd)

# Visualize Original and Transformed Structures
original_positions = parse_pdb("C:\\Amrita_S2\\DSA proj\\2c0k.pdb")
visualize_protein_structure(original_positions, transformed_positions)