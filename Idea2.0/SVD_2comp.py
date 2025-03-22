import numpy as np
from sklearn.decomposition import TruncatedSVD
import pandas as pd

# Load normalized protein coordinates
file_path = r"C:\Amrita_S2\DSA proj\Idea2.0\Normalized_Protein_Coordinates\1A3N_ca_coordinates_normalized.csv"
data = pd.read_csv(file_path)

# Extract coordinate values
coords = data[['X', 'Y', 'Z']].values

# Apply SVD to reduce to 2D
svd = TruncatedSVD(n_components=2)
reduced_coords = svd.fit_transform(coords)

# Save reduced coordinates
output_path = r"C:\Amrita_S2\DSA proj\Idea2.0\Reduced_Coordinates\1A3N_ca_coordinates_reduced.csv"
pd.DataFrame(reduced_coords, columns=['X_reduced', 'Y_reduced']).to_csv(output_path, index=False)

print(f"Reduced coordinates saved to: {output_path}")
