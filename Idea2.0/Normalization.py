import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Paths
input_directory = r"C:\Amrita_S2\DSA proj\Idea2.0\Protein_Coordinates_V1.0"
output_directory = r"C:\Amrita_S2\DSA proj\Idea2.0\Normalized_Protein_Coordinates"

# Create output folder if not exists
os.makedirs(output_directory, exist_ok=True)

# Normalize and save all proteins
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        protein_id = filename.split(".")[0]
        file_path = os.path.join(input_directory, filename)

        # Load CSV
        df = pd.read_csv(file_path, usecols=['X', 'Y', 'Z'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()  # Ensure valid numbers

        # **CHECK IF DATA IS EMPTY**
        if df.empty:
            print(f"⚠️ Warning: {protein_id} has no valid (X, Y, Z) data. Skipping...")
            continue  # Skip this file

        # Min-Max Scaling
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(df)

        # Save normalized data
        normalized_df = pd.DataFrame(normalized_data, columns=['X', 'Y', 'Z'])
        output_file_path = os.path.join(output_directory, f"{protein_id}_normalized.csv")
        normalized_df.to_csv(output_file_path, index=False)

        print(f"✅ Normalized {protein_id}, saved to {output_file_path}")

print("\n🎯 All valid protein coordinate files have been normalized and saved.")
