from Bio.PDB import PDBList, PDBParser
import csv
import os

pdb_ids = [
    "1A3N", "4HHB", "2JHD", "1BNA", "3HMM", "1UBQ", "5XNL", "2Y7Q", "6VXX", "3NIR",
    "1GFL", "4AKE", "5TQH", "1CRN", "2W5F", "3CRO", "6X2B", "1MBO", "3OXC", "5OPH",
    "2DHB", "1FME", "6LU7", "4B3R", "1D66", "2MK1", "3JQH", "4LW4", "5NZL", "3PQR",
    "2HYY", "5TPN", "1EIY", "6M0J", "3GRW", "1QYS", "2J0X", "5Y8D", "4ZTF", "6Q21",
    "1AON", "3BWD", "2FG1", "4NVH", "5HVP", "1B8O", "3FHH", "6AKY", "2QMT", "4EJZ"
] # 50 proteins added


pdbl = PDBList()

output_folder = "Protein_Coordinates_v2.0 (25-05-2025)"
os.makedirs(output_folder, exist_ok=True)

for pdb_id in pdb_ids:
    try:
        # Download and parse structure
        pdb_file = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb")
        structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_file)

        # Output file for this PDB ID
        output_file = os.path.join(output_folder, f"{pdb_id}_ca_coordinates.csv")

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["X", "Y", "Z"])  # Header

            # Extract C-alpha coordinates
            for atom in structure.get_atoms():
                if atom.get_name() == "CA":
                    writer.writerow(atom.get_coord())  # Write X, Y, Z

        print(f" {pdb_id}: Coordinates saved in '{output_file}'.")

    except Exception as e:
        print(f" {pdb_id}: Error - {e}")

print("\n All proteins processed. Each file is saved in 'Protein_Coordinates_V1.0' folder.")
