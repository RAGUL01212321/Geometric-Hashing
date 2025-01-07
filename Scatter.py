import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Function to scatter plot the preprocessed data
def scatter_plot_preprocessed_data(data_file):
    """
    Plots preprocessed data in 3D from a CSV file.
    
    Args:
        data_file (str): Path to the preprocessed data CSV file.
    """
    # Load the preprocessed data
    data = pd.read_csv(data_file)
    
    # Convert the dataframe to a NumPy array for plotting
    points = data.values
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot the points (assuming the first three columns are X, Y, Z)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', alpha=0.7)
    
    # Set labels for axes
    ax.set_title("3D Scatter Plot of Preprocessed Data")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    
    # Show the plot
    plt.show()

# Main function to load and plot the data
def main():
    data_file = "preprocessed_data.csv"  # Your preprocessed file name
    print(f"Loading preprocessed data from {data_file}...")
    
    # Call the scatter plot function
    scatter_plot_preprocessed_data(data_file)

# Run the script
if __name__ == "__main__":
    main()
