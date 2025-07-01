import argparse
import numpy as np
from PIL import Image
import io
import os

def main(input_file, output_dir):
    # Load the .npz file
    data = np.load(input_file)

    # Extract and save images
    frames_bytes = data["images_jpeg_bytes"]
    imgs_dir = os.path.join(output_dir, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    for idx, jpeg_bytes in enumerate(frames_bytes):
        img = Image.open(io.BytesIO(jpeg_bytes))
        filename = os.path.join(imgs_dir, f"{idx:05d}.png")
        img.save(filename, "PNG")

    # Save query data
    queries = data["queries_xyt"]
    query_path = os.path.join(output_dir, "queries_xyt.npy")
    np.save(query_path, queries)

    print(f"Saved {len(frames_bytes)} frames to '{imgs_dir}'")
    print(f"Saved queries to '{query_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save frames and queries from a .npz file")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    input_file = f"./{args.file}/{args.file}.npz"
    output_dir = f"./{args.file}"


    args = parser.parse_args()
    main(input_file, output_dir)
