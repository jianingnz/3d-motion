import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
import imageio.v3 as iio  # requires `imageio>=2.9`
import cv2
import argparse

def plot_3d_tracks(points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=True):
  """Visualize 3D point trajectories."""
  num_frames, num_points = points.shape[0:2]

  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  if show_occ:
    x_min, x_max = np.min(points[infront_cameras, 0]), np.max(points[infront_cameras, 0])
    y_min, y_max = np.min(points[infront_cameras, 2]), np.max(points[infront_cameras, 2])
    z_min, z_max = np.min(points[infront_cameras, 1]), np.max(points[infront_cameras, 1])
  else:
    x_min, x_max = np.min(points[visibles, 0]), np.max(points[visibles, 0])
    y_min, y_max = np.min(points[visibles, 2]), np.max(points[visibles, 2])
    z_min, z_max = np.min(points[visibles, 1]), np.max(points[visibles, 1])

  interval = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
  x_min = (x_min + x_max) / 2 - interval / 2
  x_max = x_min + interval
  y_min = (y_min + y_max) / 2 - interval / 2
  y_max = y_min + interval
  z_min = (z_min + z_max) / 2 - interval / 2
  z_max = z_min + interval

  frames = []
  for t in range(num_frames):
    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.invert_zaxis()
    ax.view_init(azim=-45)

    for i in range(num_points):
      if visibles[t, i] or (show_occ and infront_cameras[t, i]):
        color = color_map(cmap_norm(i))
        line = points[max(0, t - tracks_leave_trace) : t + 1, i]
        ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=color, linewidth=1)
        end_point = points[t, i]
        ax.scatter(xs=end_point[0], ys=end_point[2], zs=end_point[1], color=color, s=3)

    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    fig.canvas.draw()
    frames.append(canvas.buffer_rgba())
  return np.array(frames)[..., :3]

def plot_2d_tracks(video, points, visibles, infront_cameras=None, tracks_leave_trace=16, show_occ=True):
  """Visualize 2D point trajectories."""
  num_frames, num_points = points.shape[:2]

  # Precompute colormap for points
  color_map = matplotlib.colormaps.get_cmap('hsv')
  cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)
  point_colors = np.zeros((num_points, 3))
  for i in range(num_points):
    point_colors[i] = np.array(color_map(cmap_norm(i)))[:3] * 255

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  frames = []
  for t in range(num_frames):
    frame = video[t].copy()

    # Draw tracks on the frame
    line_tracks = points[max(0, t - tracks_leave_trace) : t + 1]
    line_visibles = visibles[max(0, t - tracks_leave_trace) : t + 1]
    line_infront_cameras = infront_cameras[max(0, t - tracks_leave_trace) : t + 1]
    for s in range(line_tracks.shape[0] - 1):
      img = frame.copy()

      for i in range(num_points):
        if line_visibles[s, i] and line_visibles[s + 1, i]:  # visible
          x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 2, cv2.LINE_AA)
        elif show_occ and line_infront_cameras[s, i] and line_infront_cameras[s + 1, i]:  # occluded
          x1, y1 = int(round(line_tracks[s, i, 0])), int(round(line_tracks[s, i, 1]))
          x2, y2 = int(round(line_tracks[s + 1, i, 0])), int(round(line_tracks[s + 1, i, 1]))
          cv2.line(frame, (x1, y1), (x2, y2), point_colors[i], 2, cv2.LINE_AA)

      alpha = (s + 1) / (line_tracks.shape[0] - 1)
      frame = cv2.addWeighted(frame, alpha, img, 1 - alpha, 0)

    # Draw end points on the frame
    for i in range(num_points):
      if visibles[t, i]:  # visible
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 2, point_colors[i], -1)
      elif show_occ and infront_cameras[t, i]:  # occluded
        x, y = int(round(points[t, i, 0])), int(round(points[t, i, 1]))
        cv2.circle(frame, (x, y), 2, point_colors[i], 1)

    frames.append(frame)
  frames = np.stack(frames)
  return frames

def load_video_from_images(folder_path):
    # Sorted list of .jpeg files
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    if not image_files:
        raise ValueError(f"No JPEG images found in {folder_path}")

    frames = []
    for fname in image_files:
        img_path = os.path.join(folder_path, fname)
        img = iio.imread(img_path)  # shape: (H, W, 3)
        frames.append(img)

    video = np.stack(frames)  # shape: (T, H, W, 3)
    return video


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D and 2D tracks over video.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--w",
        action="store_true",
        help="If present, save 3D tracks in world coordinates instead of camera coordinates.",
    )
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for the output videos.")
    parser.add_argument("--no_occ", action="store_true",
                        help="If set, do not show occluded points.")
    
    args = parser.parse_args()

    # === Load your predicted file ===
    if args.w:
      pred_data = np.load(f"./3d-tracks/w-{args.file}.npz")
    else:
      pred_data = np.load(f"./3d-tracks/r-{args.file}.npz")
    tracks_xyz = pred_data['tracks_XYZ']         # shape: (T, N, 3)
    visibility = pred_data['visibility']         # shape: (T, N)

    data_2dtrack = np.load(f"./co-tracker/2d-track/r-{args.file}.npz")
    tracks_xy = data_2dtrack['tracks']

    # Load image sequence
    video = load_video_from_images(f"./dataset/{args.file}/r-imgs")

    # Optional: in-front-of-camera mask (default: all visible)
    infront_cameras = np.ones_like(visibility, dtype=bool)

    # Visualizations
    video3d_viz = plot_3d_tracks(
        tracks_xyz, visibility, infront_cameras=infront_cameras, show_occ=not args.no_occ)
    video2d_viz = plot_2d_tracks(
        video, tracks_xy, visibility, infront_cameras=infront_cameras, show_occ=not args.no_occ)

    # Save output
    if args.w:
      output_dir = f"visual/w/{args.file}"
    else:
      output_dir = f"visual/c/r-{args.file}"
    os.makedirs(output_dir, exist_ok=True)
    iio.imwrite(os.path.join(output_dir, "3d_tracks_viz.mp4"), video3d_viz, fps=args.fps)
    iio.imwrite(os.path.join(output_dir, "2d_tracks_viz.mp4"), video2d_viz, fps=args.fps)

    print(f"Saved 3D track video to {output_dir}/3d_tracks_viz.mp4")
    print(f"Saved 2D track video to {output_dir}/2d_tracks_viz.mp4")


if __name__ == "__main__":
    main()

