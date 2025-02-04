import numpy as np
from Bio import PDB

def compute_centroid(atoms):
    """Calculate the centroid of given atoms"""
    coords = np.array([atom.coord for atom in atoms])
    centroid = np.mean(coords, axis=0)
    return centroid

def compute_hash_key(pdb_file):
    """Extracts centroid and radial distances from protein structure"""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    atoms = [atom for atom in structure.get_atoms()]
    centroid = compute_centroid(atoms)

    # Compute radial distances from centroid
    radial_distances = [np.linalg.norm(atom.coord - centroid) for atom in atoms[:3]]  # Selecting 3 atoms
    
    hash_key = (centroid[0], centroid[1], centroid[2], *radial_distances)
    return hash_key

# Example usage
pdb_path = "C:/Amrita_S2/DSA proj/2c0k.pdb"  # Update with your file path
hash_key = compute_hash_key(pdb_path)
print("Centroid-Based Hash Key:", hash_key)
