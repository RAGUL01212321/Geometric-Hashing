import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Load the original protein structure data
original_file = "Protein_Structure.csv"  # Update with your actual file
df = pd.read_csv(r"C:\Amrita_S2\DSA proj\Protein_SVD_Separated\3afb8cbae6c1e3fcb25f6e52780d1b3f8d7c4d48e1de651f3950180851c87ecd.csv")

# Extract the coordinate matrix (X, Y, Z values)
coordinates = df[["X", "Y", "Z"]].values

# Perform Singular Value Decomposition (SVD)
svd = TruncatedSVD(n_components=3)
U = svd.fit_transform(coordinates)  # Left singular vectors
S = svd.singular_values_            # Singular values
Vt = svd.components_                 # Right singular vectors (transposed)

# Save the SVD components for reconstruction
np.save("U_matrix.npy", U)
np.save("S_values.npy", S)
np.save("Vt_matrix.npy", Vt)

print("SVD matrices saved successfully!")