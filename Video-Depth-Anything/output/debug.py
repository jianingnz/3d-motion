import numpy as np
import matplotlib.pyplot as plt
import os

# Load the .npz file
data = np.load('drivetrack_depths.npz')
depth_array = data[list(data.files)[0]]  # Access the array (31, 854, 1280)
print(depth_array.shape)  # Should print (31, 854, 1280)

# Output folder
os.makedirs("depth_images", exist_ok=True)

# Normalize and save each frame
for i in range(depth_array.shape[0]):
    frame = depth_array[i]
    plt.imsave(f"depth_images/frame_{i:03d}.png", frame, cmap='viridis')
