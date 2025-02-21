import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the SVD-transformed CSV file
svd_file = r"C:\Amrita_S2\DSA proj\Protein_SVD_Coordinates.csv"  # Update this with your actual file path
df = pd.read_csv(svd_file)

# Create a hash table (dictionary) to store protein hashes and SVD values
hash_table = {}

# Populate hash table (Key: Hashed Protein Name, Value: SVD Coordinates)
for index, row in df.iterrows():
    protein_name = row["Protein"]  # Ensure the column exists in your CSV
    svd_values = row[["SVD1", "SVD2", "SVD3"]].tolist()  # Adjust column names if needed

    # Generate SHA-256 hash for protein name
    protein_hash = hashlib.sha256(protein_name.encode()).hexdigest()

    # Store in hash table
    hash_table[protein_hash] = svd_values

def reconstruct_structure(protein_name, svd_matrix):
    """ Reconstructs the original 3D structure using stored SVD values. """
    # Apply inverse SVD reconstruction
    U = np.array(svd_matrix)
    
    # Simulating singular values and V-transpose (Approximate)
    S = np.eye(len(U))  # Approximate singular values
    Vt = np.random.randn(len(U[0]), 3)  # Random approximation for Vt

    # Reconstruct the original coordinates
    reconstructed = np.dot(U, np.dot(S, Vt))

    # Plot the reconstructed 3D structure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c="b", marker="o")
    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(f"Reconstructed 3D Structure: {protein_name}")

    plt.show()

def retrieve_and_reconstruct(protein_name):
    """ Retrieves stored SVD representation and reconstructs the original structure. """
    protein_hash = hashlib.sha256(protein_name.encode()).hexdigest()

    if protein_hash in hash_table:
        print(f"Protein {protein_name} found! Reconstructing structure...")
        svd_matrix = np.array(hash_table[protein_hash]).reshape(-1, 3)
        reconstruct_structure(protein_name, svd_matrix)
    else:
        print("Protein not found.")

# Example Usage:
retrieve_and_reconstruct("Protein_1")  # Replace with actual protein name
