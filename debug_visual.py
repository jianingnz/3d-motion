import matplotlib.pyplot as plt  # Needed for colormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
import imageio.v3 as iio  # requires `imageio>=2.9`
import cv2
import argparse

def plot_3d_tracks(pred_points, visibles, infront_cameras=None, gt_points=None,
                   tracks_leave_trace=16, show_occ=True):
  """Visualize 3D point trajectories."""
  num_frames, num_points = pred_points.shape[0:2]

  # color_map = matplotlib.colormaps.get_cmap('hsv')
  # cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=num_points - 1)

  if infront_cameras is None:
    infront_cameras = np.ones_like(visibles).astype(bool)

  if show_occ:
    x_min, x_max = np.min(gt_points[infront_cameras, 0]), np.max(gt_points[infront_cameras, 0])
    y_min, y_max = np.min(gt_points[infront_cameras, 2]), np.max(gt_points[infront_cameras, 2])
    z_min, z_max = np.min(gt_points[infront_cameras, 1]), np.max(gt_points[infront_cameras, 1])
  else:
    x_min, x_max = np.min(gt_points[visibles, 0]), np.max(gt_points[visibles, 0])
    y_min, y_max = np.min(gt_points[visibles, 2]), np.max(gt_points[visibles, 2])
    z_min, z_max = np.min(gt_points[visibles, 1]), np.max(gt_points[visibles, 1])

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
    ax.view_init(azim=45)

    for i in range(num_points):
      if visibles[t, i] or (show_occ and infront_cameras[t, i]):
        color = 'red'
        line = pred_points[max(0, t - tracks_leave_trace) : t + 1, i]
        ax.plot(xs=line[:, 0], ys=line[:, 2], zs=line[:, 1], color=color, linewidth=1)
        end_point = pred_points[t, i]
        ax.scatter(xs=end_point[0], ys=end_point[2], zs=end_point[1], color=color, s=3)
    if gt_points is not None:
      for i in range(gt_points.shape[1]):
        gt_color = 'green'  # or fixed RGB or different colormap
        gt_line = gt_points[max(0, t - tracks_leave_trace): t + 1, i]
        ax.plot(xs=gt_line[:, 0], ys=gt_line[:, 2], zs=gt_line[:, 1],
                color=gt_color, linewidth=1)
        end_point = gt_points[t, i]
        ax.scatter(xs=end_point[0], ys=end_point[2], zs=end_point[1],
                    color=gt_color, s=2)

    fig.subplots_adjust(left=-0.05, right=1.05, top=1.05, bottom=-0.05)
    fig.canvas.draw()
    frames.append(canvas.buffer_rgba())
  return np.array(frames)[..., :3]

def umeyama_alignment(pred_xyz, gt_xyz):
    """
    Align predicted 3D tracks to GT using Umeyama: scale + rotation + translation.
    
    Args:
        pred_xyz: (T, N, 3) predicted trajectory
        gt_xyz: (T, N, 3) ground-truth trajectory

    Returns:
        pred_aligned: (T, N, 3) aligned predicted trajectory
        (R, t, s): rotation matrix (3x3), translation (3,), scale (float)
    """
    X = pred_xyz.reshape(-1, 3)
    Y = gt_xyz.reshape(-1, 3)

    # Mask valid entries
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[mask]
    Y = Y[mask]

    # Means
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    # Centered
    Xc = X - mu_X
    Yc = Y - mu_Y

    # Covariance
    cov = Xc.T @ Yc / len(X)

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / np.sum(Xc ** 2)
    t = mu_Y - s * R @ mu_X

    # Apply transformation
    pred_aligned = (s * (R @ pred_xyz.reshape(-1, 3).T).T + t).reshape(pred_xyz.shape)
    return pred_aligned, (R, t, s)



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
      pred_data = np.load(f"./3d-tracks/{args.file}.npz")
    tracks_xyz = pred_data['tracks_XYZ']         # shape: (T, N, 3)
    visibility = pred_data['visibility']         # shape: (T, N)

    # Optional: in-front-of-camera mask (default: all visible)
    infront_cameras = np.ones_like(visibility, dtype=bool)

    # gt_data = np.load(f"./dataset/{args.file}/{args.file}.npz")
    gt_data = np.load(f"./3d-tracks/r-{args.file}.npz")
    gt_tracks_xyz = gt_data['tracks_XYZ']  # shape: (T, N, 3)

    # pred_norms = np.linalg.norm(tracks_xyz, axis=-1).flatten()
    # gt_norms = np.linalg.norm(gt_tracks_xyz, axis=-1).flatten()
    # valid = (pred_norms > 1e-8) & np.isfinite(pred_norms) & np.isfinite(gt_norms)
    # scale = np.median(gt_norms[valid] / pred_norms[valid])
    # tracks_xyz = tracks_xyz * scale
    # tracks_xyz, (R, t, s) = umeyama_alignment(tracks_xyz, gt_tracks_xyz)
    # print(f"Applied Umeyama alignment: scale={s:.4f}, rotation=\n{R}, translation={t}")

    # Visualizations
    video3d_viz = plot_3d_tracks(
      tracks_xyz, visibility, infront_cameras=infront_cameras,
      gt_points=gt_tracks_xyz, show_occ=not args.no_occ)
    
    # Save output
    if args.w:
      output_dir = f"visual/w/{args.file}"
    else:
      output_dir = f"visual/{args.file}"
    os.makedirs(output_dir, exist_ok=True)
    iio.imwrite(os.path.join(output_dir, "3d_tracks_viz.mp4"), video3d_viz, fps=args.fps)

    print(f"Saved 3D track video to {output_dir}/3d_tracks_viz.mp4")


if __name__ == "__main__":
    main()

