import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_plot_preprocessed_data(data_file):
    
    data=pd.read_csv(data_file)
    points = data.values
    
    # 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', alpha=0.7)
    
    # Set labels for axes
    ax.set_title("3D Scatter Plot of Preprocessed Data")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.show()

# Main function to load and plot the data
data_file=r"C:\Amrita_S2\DSA proj\1A3N_ca_coordinates_preprocessed.csv"
print(f"Loading preprocessed data from {data_file}...")    
scatter_plot_preprocessed_data(data_file)
