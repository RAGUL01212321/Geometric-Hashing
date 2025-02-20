import pandas as pd
file_path=r"C:\Amrita_S2\DSA proj\Protein_Coordinates_V1.0\1A3N_ca_coordinates.csv"
df = pd.read_csv(file_path)

x_mean, y_mean, z_mean = df["X"].mean(), df["Y"].mean(), df["Z"].mean() # Mean

# Centering 
df["X_centered"] = df["X"] - x_mean
df["Y_centered"] = df["Y"] - y_mean
df["Z_centered"] = df["Z"] - z_mean

# Min and Max coordinates
x_min, x_max = df["X_centered"].min(), df["X_centered"].max()
y_min, y_max = df["Y_centered"].min(), df["Y_centered"].max()
z_min, z_max = df["Z_centered"].min(), df["Z_centered"].max()

# Normalize ( b/w 0 to 1)
df["X_norm"] = (df["X_centered"] - x_min) / (x_max - x_min)
df["Y_norm"] = (df["Y_centered"] - y_min) / (y_max - y_min)
df["Z_norm"] = (df["Z_centered"] - z_min) / (z_max - z_min)

df_final=df[["X_norm","Y_norm","Z_norm"]]
df_final.to_csv("processed_protein_coordinates.csv", index=False)
print("Centering & Normalization complete! Saved as 'processed_protein_coordinates.csv'")