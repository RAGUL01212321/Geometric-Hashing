import pandas as pd
import numpy as np
import os
import hashlib

# Load the SVD-transformed CSV file
svd_file = r"C:\Amrita_S2\DSA proj\Protein_Coordinates_v2.0 (25-05-2025)\1A3N_ca_coordinates.csv"  # Update with your actual file
df = pd.read_csv(svd_file)

# Create a directory to store individual protein files
output_dir = "Protein_SVD_Separated"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store hash mappings
hash_mapping = {}

# Iterate over unique proteins
for protein in df["Protein"].unique():
    # Extract SVD coordinates for the current protein
    protein_data = df[df["Protein"] == protein][["SVD1", "SVD2", "SVD3"]].values

    # Compute Centroid (Mean of all SVD coordinates)
    centroid = np.mean(protein_data, axis=0)

    # Generate a Hash Key (SHA-256 of centroid values)
    hash_key = hashlib.sha256(centroid.tobytes()).hexdigest()

    # Save the SVD coordinates to a separate file named after the hash key
    protein_filename = f"{hash_key}.csv"
    np.savetxt(os.path.join(output_dir, protein_filename), protein_data, delimiter=",")

    # Store mapping (Protein Name → Hash Key)
    hash_mapping[protein] = hash_key

# Save the mapping file for retrieval
mapping_df = pd.DataFrame(list(hash_mapping.items()), columns=["Protein", "Hash_Key"])
mapping_df.to_csv("Protein_Hash_Mapping.csv", index=False)

print(f"Processed {len(hash_mapping)} proteins. Data saved in '{output_dir}'.")
print("Protein hash mapping saved as 'Protein_Hash_Mapping.csv'.")

