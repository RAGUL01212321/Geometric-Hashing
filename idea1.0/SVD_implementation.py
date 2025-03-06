import numpy as np
import pandas as pd
from scipy.linalg import svd

# Load the preprocessed C-alpha coordinates CSV
file_path = r"C:\Amrita_S2\DSA proj\1A3N_ca_coordinates_preprocessed.csv"
data = pd.read_csv(file_path)

# Extract only the last 3 columns (X, Y, Z coordinates)
coords = data.iloc[:, -3:].values  # Assuming the last 3 columns are the normalized coordinates

# Apply SVD
U, S, Vt = svd(coords, full_matrices=False)

# Choose top-k singular vectors (dimensionality reduction)
k = 3  # You can change this based on how much information you want to retain
V_k = Vt[:k, :]  # Keep the top k singular vectors

# Transform original data to reduced representation
X_reduced = np.dot(coords, V_k.T)  # Project onto new space

# Save reduced features
output_file = r"C:\Amrita_S2\DSA proj\Protein_Coordinates_V1.0\1A3N_svd_features.csv"
pd.DataFrame(X_reduced).to_csv(output_file, index=False)

print(f"SVD applied! Reduced features saved to {output_file}")
