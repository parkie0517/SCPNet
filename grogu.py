import numpy as np

def fuse_multi_scan(points, pose0, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))  # Add a homogeneous coordinate
    print("Step 1 - Homogeneous points (hpoints):\n", hpoints)
    print(hpoints.shape)
    exit(0)
    
    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
    print("Step 2 - Transformed points (new_points):\n", new_points)
    
    new_points = new_points[:, :3]
    print("Step 3 - New points without homogeneous coordinate:\n", new_points)
    
    new_coords = new_points - pose0[:3, 3]
    print("Step 4 - New coordinates after subtracting pose0 translation:\n", new_coords)
    
    new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
    print("Step 5 - New coordinates after applying pose0 rotation:\n", new_coords)
    
    new_coords = np.hstack((new_coords, points[:, 3:]))
    print("Step 6 - Final coordinates with original attributes:\n", new_coords)
    
    return new_coords

# Generate 10 random 3D points with an additional attribute (e.g., intensity)
points = np.random.rand(3, 4)

# Create random poses (4x4 transformation matrices)
pose0 = np.eye(4)  # Identity matrix as pose0
pose = np.random.rand(4, 4)  # Random pose

# Normalize the poses to make them valid rotation matrices with translations
pose[:3, :3] = np.linalg.qr(np.random.rand(3, 3))[0]
pose[:3, 3] = np.random.rand(3)

# Call the function and show the transformations
transformed_points = fuse_multi_scan(points, pose0, pose)
print("Transformed points:\n", transformed_points)