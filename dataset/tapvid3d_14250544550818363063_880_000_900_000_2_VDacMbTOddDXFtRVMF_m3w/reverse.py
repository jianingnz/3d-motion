import os
import shutil
from pathlib import Path

# Paths
src_folder = Path("./imgs")     # Replace with your actual path
dst_folder = Path("./r-imgs")     # Replace with your desired output path

# Create destination folder
dst_folder.mkdir(parents=True, exist_ok=True)

# Get sorted list of source images
image_files = sorted(src_folder.glob("*.png"))

# Reverse the list
image_files_reversed = list(reversed(image_files))

# Save to new folder with same zero-padded format
num_digits = len(image_files[0].stem)  # e.g., 5 for '00000'
for idx, img_path in enumerate(image_files_reversed):
    new_name = f"{idx:0{num_digits}d}.png"
    shutil.copy(img_path, dst_folder / new_name)

print(f"Reversed {len(image_files)} images to: {dst_folder}")