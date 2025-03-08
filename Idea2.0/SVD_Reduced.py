import numpy as np
import pandas as pd
import os
from sklearn.decomposition import TruncatedSVD

# Load the feature matrix
input_csv = r"C:\Amrita_S2\DSA proj\Idea2.0\Feature_Matrices\protein_feature_matrix.csv"
df = pd.read_csv(input_csv)

# Convert to NumPy array (excluding header if necessary)
feature_matrix = df.values

# Apply SVD for dimensionality reduction
num_components = 100  # Adjust this value as needed
svd = TruncatedSVD(n_components=num_components, random_state=42)
reduced_features = svd.fit_transform(feature_matrix)

# Save the reduced feature matrix to a new CSV file
output_folder = "Reduced_Features"
os.makedirs(output_folder, exist_ok=True)
output_csv = os.path.join(output_folder, "reduced_feature_matrix.csv")

# Convert back to DataFrame for saving
reduced_df = pd.DataFrame(reduced_features)
reduced_df.to_csv(output_csv, index=False)

print(f"Reduced feature matrix saved to: {output_csv}")
