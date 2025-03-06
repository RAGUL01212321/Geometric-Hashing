import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Generate random 2D points (representing simplified protein structure features)
np.random.seed(42)
points = np.random.rand(20, 2)  # 20 points in 2D space

# Build KD-Tree
tree = KDTree(points)

# Function to plot KD-Tree partitions
def plot_kdtree(points, depth=0, xlim=(0, 1), ylim=(0, 1)):
    if len(points) == 0:
        return
    k = 2  # Dimensions
    axis = depth % k  # Alternate between x (0) and y (1)
    
    points = points[np.argsort(points[:, axis])]  # Sort points by splitting axis
    median_idx = len(points) // 2
    median_point = points[median_idx]
    
    # Plot partition lines
    if axis == 0:
        plt.axvline(median_point[0], ymin=ylim[0], ymax=ylim[1], color='r', linestyle="--")
    else:
        plt.axhline(median_point[1], xmin=xlim[0], xmax=xlim[1], color='b', linestyle="--")
    
    # Recursively plot left and right partitions
    plot_kdtree(points[:median_idx], depth + 1, (xlim[0], median_point[0]), ylim)
    plot_kdtree(points[median_idx + 1:], depth + 1, (median_point[0], xlim[1]), ylim)

# Plot points
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color='black', label="Protein Features")

# Draw KD-Tree partition lines
plot_kdtree(points)

plt.title("KD-Tree Partitioning (2D Example)")
plt.legend()
plt.show()
