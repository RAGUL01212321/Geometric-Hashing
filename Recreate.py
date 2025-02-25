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
