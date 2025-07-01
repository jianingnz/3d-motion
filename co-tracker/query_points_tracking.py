import os
import argparse
import torch
import numpy as np
import imageio.v3 as iio
from PIL import Image
import glob
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["video", "frames"],
        required=True,
        help="Inference mode: 'video' for video file or 'frames' for folder of frames",
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/scaled_offline.pth",
        help="Path to CoTracker model checkpoint",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    if args.mode == "frames":
        video_path = f"/home/jianing/research/3d-motion/dataset/{args.file}/imgs"
    else:
        video_path = f"/home/jianing/research/3d-motion/dataset/{args.file}/r-{args.file}.mp4"

    queries_path = f"/home/jianing/research/3d-motion/dataset/{args.file}/queries_xyt.npy"

    # Load video
    if args.mode == "video":
      frames = iio.imread(video_path, plugin="FFMPEG")
    else:
      frame_paths = sorted(glob.glob(os.path.join(video_path, '*')))
      frames = [torch.tensor(np.array(Image.open(p))) for p in frame_paths]
      frames = torch.stack(frames).numpy()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    # Load CoTracker
    cotracker = CoTrackerPredictor(checkpoint=args.checkpoint).to(device)

    # Manual query points [frame_index, x, y]
    queries_np = np.load(queries_path)  # [x, y, t]
    num_frames = video.shape[1]
    print(f"Total number of frames of the video: {num_frames}")
    # print(queries_np[0])
    queries_np[:, 2] = (num_frames - 1) - queries_np[:, 2]
    # print(queries_np[0])

    queries = torch.tensor(
       queries_np[:, [2, 0, 1]],  # [t, x, y]
       dtype=torch.float32,
       device=device
    )

    # Predict tracks
    pred_tracks, pred_visibility = cotracker(video, queries=queries[None])
    print(f"track shape: {pred_tracks.shape}")
    print(f"visibility shape: {pred_visibility.shape}")

    tracks_np = pred_tracks.cpu().numpy().squeeze(0)
    visibility_np = pred_visibility.cpu().numpy().squeeze(0)
    video_np = video.cpu().numpy().squeeze(0)
    video_dim =  (video_np.shape[2], video_np.shape[3])
    print(f"np_track shape: {tracks_np.shape}")
    print(f"np_visibility shape: {visibility_np.shape}")
    print(f"video_dim shape: {video_dim}")

    # Save both in a single .npz file
    np.savez(f"./2d-track/r-{args.file}.npz", tracks=tracks_np, visibility=visibility_np, dim=video_dim)

    # Visualize
    vis = Visualizer(
        save_dir='./videos',
        linewidth=3,
        mode='rainbow',
        tracks_leave_trace=-1,
        pad_value=100
    )
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename="r-"+args.file
    )
    # visualize_2d_tracks_on_video(video, pred_tracks.squeeze(0), pred_visibility.squeeze(0), "./videos/drivetrack")



if __name__ == "__main__":
    main()
