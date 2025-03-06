import numpy as np
import pandas as pd

# Load the CSV file (Update path if needed)
file_path = r"C:\Amrita_S2\DSA proj\Protein_Coordinates_v2.0 (25-05-2025)\1A3N_ca_coordinates.csv"
df = pd.read_csv(file_path)

# Convert X, Y, Z into a NumPy matrix
protein_matrix = df.to_numpy()

# Step 1: Compute SVD
U, S, Vt = np.linalg.svd(protein_matrix, full_matrices=False)

# Convert singular values into a diagonal matrix
S_diag = np.diag(S)

# Step 2: Reconstruct the matrix from SVD
reconstructed_matrix = np.dot(U, np.dot(S_diag, Vt))

# Step 3: Compare Original vs. Reconstructed
print("Original Protein Coordinates:\n", protein_matrix)
print("\nReconstructed Coordinates:\n", reconstructed_matrix)
print("\nDifference (Error):\n", protein_matrix - reconstructed_matrix)

# Optional: Save the reconstructed coordinates for analysis
output_path = r"C:\Amrita_S2\DSA proj\Reconstructed_Protein_Coordinates.csv"
pd.DataFrame(reconstructed_matrix, columns=['X', 'Y', 'Z']).to_csv(output_path, index=False)
print("\nReconstructed file saved at:", output_path)
