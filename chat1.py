import numpy as np
from scipy.spatial import KDTree
from Bio.PDB import PDBParser
import hashlib

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
    return np.array(invariants)

# Step 3: Hashing Function for Efficient Retrieval
def generate_hash(feature_vector):
    feature_str = "_".join(map(str, feature_vector.flatten()))
    return hashlib.md5(feature_str.encode()).hexdigest()

# Step 4: Store & Retrieve Structures Efficiently
class ProteinDatabase:
    def __init__(self):
        self.feature_hashes = {}  # Dictionary for hashing
        self.kd_tree = None
        self.features = []

    def add_protein(self, pdb_file, protein_id):
        positions = parse_pdb(pdb_file)
        features = compute_invariants(positions)
        
        for feature in features:
            hash_key = generate_hash(feature)
            self.feature_hashes[hash_key] = protein_id
            self.features.append(feature)
        
        self.kd_tree = KDTree(self.features)  # Fast retrieval

    def retrieve_protein(self, query_feature):
        if self.kd_tree is None:
            return None
        
        _, idx = self.kd_tree.query(query_feature)
        hash_key = generate_hash(self.features[idx])
        return self.feature_hashes.get(hash_key, "Not Found")

# Example Usage
db = ProteinDatabase()
db.add_protein("C:\\Amrita_S2\\DSA proj\\2c0k.pdb", "Protein_1")

query_feature = compute_invariants(parse_pdb("C:\\Amrita_S2\\DSA proj\\2c0k.pdb"))[0]
retrieved_protein = db.retrieve_protein(query_feature)
print("Retrieved Protein:", retrieved_protein)
