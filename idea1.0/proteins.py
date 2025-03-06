from Bio.PDB import PDBList

# List of protein PDB IDs to download
pdb_ids = ["2c0k", "1a3n", "4hhb", "1crn", "2bfu"]

# Initialize PDB downloader
pdbl = PDBList()

# Download PDB files and store in a directory
for pdb_id in pdb_ids:
    pdbl.retrieve_pdb_file(pdb_id, pdir="C:\\Amrita_S2\\DSA_proj\\pdb_files", file_format="pdb")
