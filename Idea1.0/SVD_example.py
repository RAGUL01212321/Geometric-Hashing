import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random 3D data (simulating protein structure points)
np.random.seed(42)
X = np.random.randn(100, 3)  # 100 points in 3D

# Apply Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Reduce dimensions (keep only the first 2 components)
X_reduced = U[:, :2] @ np.diag(S[:2])

# Plot Original 3D Data
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], color='blue', label="Original Data")
ax1.set_title("Original 3D Data")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("X3")
ax1.legend()

# Plot Reduced 2D Data
ax2 = fig.add_subplot(122)
ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], color='red', label="SVD Reduced Data")
ax2.set_title("SVD Reduced 2D Data")
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")
ax2.legend()

plt.show()
#