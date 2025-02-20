import random

def generate_3d_points(n):
    points = []
    for _ in range(n):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        z = random.uniform(-10, 10)
        points.append([x, y, z])
    return points


def mean_vector(points):
    n = len(points)
    mean = [sum(p[i] for p in points) / n for i in range(3)]
    return mean

def covariance_matrix(points, mean):
    n = len(points)
    cov_matrix = [[0] * 3 for _ in range(3)]
    
    for p in points:
        for i in range(3):
            for j in range(3):
                cov_matrix[i][j] += (p[i] - mean[i]) * (p[j] - mean[j])
    
    for i in range(3):
        for j in range(3):
            cov_matrix[i][j] /= (n - 1)
    
    return cov_matrix

# Step 4: Power Iteration to find the dominant eigenvector
def power_iteration(matrix, num_iters=100):
    vec = [1, 1, 1]  # Initial guess
    for _ in range(num_iters):
        # Matrix-vector multiplication
        new_vec = [sum(matrix[i][j] * vec[j] for j in range(3)) for i in range(3)]
        # Normalize the vector
        mag = sum(v**2 for v in new_vec) ** 0.5
        vec = [v / mag for v in new_vec]
    
    return vec

# Step 5: Project the points onto the first two principal components
def project_points(points, mean, eigenvectors):
    projected = []
    for p in points:
        # Center the data
        centered = [p[i] - mean[i] for i in range(3)]
        # Project onto eigenvectors
        new_x = sum(centered[i] * eigenvectors[0][i] for i in range(3))
        new_y = sum(centered[i] * eigenvectors[1][i] for i in range(3))
        projected.append((new_x, new_y))
    return projected

# Generate 3D points
points = generate_3d_points(100)

# Compute mean and covariance matrix
mean = mean_vector(points)
cov_matrix = covariance_matrix(points, mean)

# Find first two principal components using power iteration
eigenvec1 = power_iteration(cov_matrix)
# Defining a second vector perpendicular to the first (basic approach)
eigenvec2 = [-eigenvec1[1], eigenvec1[0], 0]  

# Project points onto 2D plane
projected_points = project_points(points, mean, [eigenvec1, eigenvec2])

# Print some projected points
for i in range(10):
    print(f"Original: {points[i]}, Projected: {projected_points[i]}")
