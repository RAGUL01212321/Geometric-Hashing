import numpy as np
import pandas as pd

# Load the original protein structure dataset
original_file = r"C:\Amrita_S2\DSA proj\Protein_Coordinates_v2.0 (25-05-2025)\1AON_ca_coordinates.csv"  # Update this with your actual file
df = pd.read_csv(original_file)

# Perform SVD
U, S, Vt = np.linalg.svd(df.values, full_matrices=False)

# Save SVD components for later use
np.save("U_matrix.npy", U)
np.save("S_matrix.npy", S)
np.save("Vt_matrix.npy", Vt)

print("SVD matrices saved successfully!")

# Load the saved SVD matrices
U = np.load("U_matrix.npy")
S = np.load("S_matrix.npy")
Vt = np.load("Vt_matrix.npy")

# Convert singular values into a diagonal matrix
S_matrix = np.diag(S)

# Reconstruct the original dataset
reconstructed_data = np.dot(U, np.dot(S_matrix, Vt))

# Save the reconstructed structure
np.savetxt("Reconstructed_Protein_Structure.csv", reconstructed_data, delimiter=",")

print("Reconstruction complete! Data saved as 'Reconstructed_Protein_Structure.csv'.")
