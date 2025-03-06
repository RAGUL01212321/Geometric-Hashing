import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# SVD comparision

# Path to your PDB file
pdb_file_path = r"C:\Amrita_S2\DSA proj\2c0k.pdb"  

# Function to extract 3D atomic coordinates from PDB file
def extract_coordinates(pdb_path):
    coordinates = []
    with open(pdb_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):  # Select only atom lines
                x = float(line[30:38].strip())  # X coordinate
                y = float(line[38:46].strip())  # Y coordinate
                z = float(line[46:54].strip())  # Z coordinate
                coordinates.append([x, y, z])
    return np.array(coordinates)

# Load protein structure from PDB
protein_structure = extract_coordinates(pdb_file_path)

# Apply Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(protein_structure, full_matrices=False)

# Reduce to 2D (keeping top 2 singular values)
S_reduced = np.diag(S)
reduced_structure = np.dot(U[:, :2], S_reduced[:2, :2])

# Plot Original 3D Protein Structure
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(protein_structure[:, 0], protein_structure[:, 1], protein_structure[:, 2], c='blue', label="Original")
ax1.set_title("Original 3D Protein Structure")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Plot SVD-Reduced 2D Structure
ax2 = fig.add_subplot(122)
ax2.scatter(reduced_structure[:, 0], reduced_structure[:, 1], c='red', label="SVD-Reduced")
ax2.set_title("SVD-Reduced Protein Structure (Projected to 2D)")
ax2.set_xlabel("Component 1")
ax2.set_ylabel("Component 2")
ax2.legend()

plt.show()