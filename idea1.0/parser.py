# Import necessary libraries
from Bio.PDB import PDBParser
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
pdb_file="2c0k.pdb"
# Function to parse PDB file and extract atomic coordinates
def extract_coordinates(pdb_file):  
    """
    Parses a PDB file and extracts atomic coordinates.
    
    Args:
        pdb_file (str): Path to the PDB file.
    
    Returns:
        np.ndarray: Array of atomic coordinates (x, y, z).
    """
    parser = PDBParser(QUIET=True)  # Initialize PDB parser
    structure = parser.get_structure("Protein", "2c0k.pdb")
    
    # Extract coordinates of all atoms
    coordinates = []
    for atom in structure.get_atoms():
        coordinates.append(atom.coord)
    
    return np.array(coordinates)

# Function to calculate pairwise distances (distance matrix)
def calculate_distance_matrix(coordinates):
    """
    Calculates the pairwise distance matrix for given coordinates.
    
    Args:
        coordinates (np.ndarray): Array of atomic coordinates (x, y, z).
    
    Returns:
        np.ndarray: Pairwise distance matrix.
    """
    return squareform(pdist(coordinates))

# Function to preprocess the distance matrix using PCA
def preprocess_distance_matrix(distance_matrix, n_components=3):
    """
    Reduces the dimensionality of the distance matrix using PCA.
    
    Args:
        distance_matrix (np.ndarray): Pairwise distance matrix.
        n_components (int): Number of principal components to retain.
    
    Returns:
        np.ndarray: Reduced representation of the distance matrix.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(distance_matrix)
    return reduced_data

# Function to save preprocessed data to a CSV file
def save_to_csv(data, output_file):
    """
    Saves the preprocessed data to a CSV file.
    
    Args:
        data (np.ndarray): Preprocessed data.
        output_file (str): Path to save the CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

# Main function to execute the pipeline
def main():
    # Path to the PDB file (replace with your file path)
    pdb_file = "example.pdb"  # Replace with the path to your downloaded PDB file
    
    # Extract atomic coordinates
    print("Extracting atomic coordinates from PDB file...")
    coordinates = extract_coordinates("2c0k.pdb")
    print(f"Extracted {len(coordinates)} atomic coordinates.")
    
    # Calculate pairwise distance matrix
    print("Calculating pairwise distance matrix...")
    distance_matrix = calculate_distance_matrix(coordinates)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Preprocess distance matrix using PCA
    print("Reducing dimensionality using PCA...")
    preprocessed_data = preprocess_distance_matrix(distance_matrix)
    print(f"Reduced data shape: {preprocessed_data.shape}")
    
    # Save preprocessed data to CSV
    output_file = "preprocessed_data.csv"
    save_to_csv(preprocessed_data, output_file)

    print("Pipeline completed successfully!")

# Run the pipeline
if __name__ == "__main__":
    main()
