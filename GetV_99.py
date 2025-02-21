import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

num_proteins = 10
atoms_per_protein = 50
protein_names = [f"Protein_{i+1}" for i in range(num_proteins)]

protein_data = []
svd_matrices = {}

for protein in protein_names:
    x = np.random.uniform(-10, 10, atoms_per_protein)
    y = np.random.uniform(-10, 10, atoms_per_protein)
    z = np.random.uniform(-10, 10, atoms_per_protein)
    
    coords = np.column_stack((x, y, z))
    
    # Apply SVD
    U, S, Vt = np.linalg.svd(coords, full_matrices=False)

    # Store matrices for reconstruction
    svd_matrices[protein] = {"U": U, "S": S, "Vt": Vt}

    # Store reduced representation
    for i in range(atoms_per_protein):
        protein_data.append([protein, U[i, 0] * S[0], U[i, 1] * S[1], U[i, 2] * S[2]])

df = pd.DataFrame(protein_data, columns=["Protein", "SVD1", "SVD2", "SVD3"])
df.to_csv("Protein_SVD_Coordinates.csv", index=False)

# Save U, S, Vt for reconstruction
np.save("svd_matrices.npy", svd_matrices)

print("✅ SVD data and matrices saved successfully!")
