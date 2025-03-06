import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

# Define file paths
input_csv = r"C:\Amrita_S2\DSA proj\Idea2.0\Normalized_Protein_Coordinates\1A3N_ca_coordinates_normalized.csv"  # Replace with your normalized CSV filename
output_folder = "Feature_Matrices"
output_csv = os.path.join(output_folder, "protein_feature_matrix.csv")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load normalized protein data
df = pd.read_csv(input_csv)

# Check if required columns exist
if not all(col in df.columns for col in ["X", "Y", "Z"]):
    raise ValueError("CSV file must contain 'X', 'Y', and 'Z' columns.")

# Convert to NumPy array
coordinates = df[["X", "Y", "Z"]].values

# Compute pairwise Euclidean distances (alternative features: angles, torsions, etc.)
distance_matrix = squareform(pdist(coordinates, metric='euclidean'))

# Convert to DataFrame for saving
feature_df = pd.DataFrame(distance_matrix)

# Save to CSV
feature_df.to_csv(output_csv, index=False)

print(f"Feature matrix saved at: {output_csv}")
