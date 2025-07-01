import time
from pathlib import Path

import cv2
import numpy as np
import argparse
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps

import viser
import viser.transforms as tf


class MegaSamLoader:
    def __init__(self, data_path: Path, tracks_path: Path, depth_path: Path):
        # Read metadata.
        data = np.load(data_path)

        depth = np.load(depth_path)['depths']
        T, H, W = depth.shape
        h, w = data["images"].shape[1], data["images"].shape[2]

        resized_depth = np.empty((T, h, w), dtype=depth.dtype)
        for i in range(T):
            resized_depth[i] = cv2.resize(depth[i], (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Change from disparity value to real depth value
        resized_depth = np.max(resized_depth)-resized_depth

        # mean_orig = data["depths"].mean()
        # print(mean_orig)
        # mean_resized = resized_depth.mean()
        # print(mean_resized)

        # scale = mean_orig / (mean_resized + 1e-8)
        # print(scale)
        # print(resized_depth.shape)
        # print(resized_depth[0])
        # resized_depth *= scale

        # resized_depth_normal = (resized_depth - resized_depth.min()) / (resized_depth.max() - resized_depth.min())
        # resized_depth = 1.0 - resized_depth_normal
        # resized_depth = resized_depth * (data["depths"].max()-data["depths"].min()) + data["depths"].min()

        min_d = np.min(resized_depth)
        max_d = np.max(resized_depth)
        depths_normalized = (resized_depth - min_d) / (max_d - min_d + 1e-8)

        # print(resized_depth.shape)
        # print(resized_depth[0])

        # print(data["depths"].shape)
        # print(data["depths"][0])

        self.K: np.ndarray = data["intrinsic"]
        self.K_inv = np.linalg.inv(self.K)
        # self.rgbd = np.concatenate([data["images"].astype(np.float16) / 255, data["depths"][..., None]], axis=-1)
        self.rgbd = np.concatenate([data["images"].astype(np.float16) / 255, depths_normalized[..., None]], axis=-1)
        self.cam_c2w = data["cam_c2w"]

        # Load 2D tracks
        tracks_data = np.load(tracks_path)
        self.tracks = tracks_data["tracks"]           # (num_frames, num_tracks, 2)
        self.visibility = tracks_data["visibility"]   # (num_frames, num_tracks)
        self.dimension = tracks_data["dim"]

    @property
    def num_frames(self) -> int:
        return self.rgbd.shape[0]

    @property
    def height(self) -> int:
        return self.rgbd.shape[1]

    @property
    def width(self) -> int:
        return self.rgbd.shape[2]

    def get_points(self, index: int, downsample_factor: int = 1) -> tuple[np.ndarray, np.ndarray]:
        rgbd = self.rgbd[index]
        cam_c2w = self.cam_c2w[index]
        K_inv = self.K_inv
        h, w = rgbd.shape[:2]
        depth = rgbd[..., 3]
        img_coords = np.concatenate([
            np.mgrid[:w, :h].transpose((2, 1, 0)).astype(np.float16),
            depth[..., None]
        ], axis=-1)  # H,W,3
        if downsample_factor > 1:
            img_coords = img_coords[::downsample_factor, ::downsample_factor]
            rgbd = rgbd[::downsample_factor, ::downsample_factor]
        # print("shape of img_coords:", img_coords.shape)
        cam_coords = img_coords @ K_inv.T
        # print("shape of cam_coords:", cam_coords.shape)
        cam_coords = np.concatenate([
            cam_coords, np.ones((*img_coords.shape[:2], 1), dtype=np.float16)
        ], axis=-1)
        # print("shape of cam_coords:", cam_coords.shape)
        cam_coords = cam_coords @ cam_c2w.T
        # print("shape of cam_coords:", cam_coords.shape)
        cam_coords = cam_coords[..., :3] / cam_coords[..., 3:]
        # print("shape of cam_coords:", cam_coords.shape)
        cam_coords = cam_coords.reshape(-1, 3)
        # print("shape of world_coords:", cam_coords.shape)
        rgb = rgbd[..., :3].reshape(-1, 3)
        return cam_coords, rgb
    
    def get_tracks(self, index: int, world_track: bool) -> tuple[np.ndarray]:
        rgbd = self.rgbd[index]
        cam_c2w = self.cam_c2w[index]
        K_inv = self.K_inv
        h, w = rgbd.shape[:2]
        depth = rgbd[..., 3]
        H, W = self.dimension

        track_coords = self.tracks[index]  # (N, 2)
        track_visibility = self.visibility[index].astype(bool)  # (N,)
        num_tracks = track_coords.shape[0]  # N
        # print("total track numbers:", num_tracks)
        # print("original coords", track_coords[:5])

        # transform the coordinates
        scale_y = h / H
        scale_x = w / W
        scaled_coords = track_coords.copy()
        scaled_coords[:, 0] *= scale_x  # x (width)
        scaled_coords[:, 1] *= scale_y  # y (height)
        # print("afterwards", scaled_coords[:5])

        # print("visibles", visible_coords[:5])
        pixel_coords = np.round(scaled_coords).astype(int)
        # print("for depth", pixel_coords[:5])
        # Clip to image boundaries to avoid indexing errors
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, w - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, h - 1)

        # Access depth values at these coordinates (note y = row, x = col)
        depth_values = depth[pixel_coords[:, 1], pixel_coords[:, 0]]
        # print(depth_values.shape)  # (N,)
        visible_depths = depth_values[track_visibility]
        if visible_depths.size > 0:
            mean_depth = visible_depths.mean()
        else:
            mean_depth = 1.0
        depth_values[~track_visibility] = mean_depth


        coord_depth = np.concatenate([scaled_coords, depth_values[:, None]], axis=1)
        # print(coord_depth.shape)
        cam_coords = coord_depth @ K_inv.T

        if world_track:
        # print(cam_coords.shape)
            cam_coords = np.concatenate([
                cam_coords, np.ones((*coord_depth.shape[:1], 1), dtype=np.float16)
            ], axis=-1)
        # # print(cam_coords.shape)
            world_coords = cam_coords @ cam_c2w.T
        # # print(world_coords.shape)
            world_coords = world_coords[..., :3] / world_coords[..., 3:]
        # print(world_coords.shape)
            return world_coords
        else:
            return cam_coords

def main(
    data_path: Path,
    track_path: Path,
    depth_path: Path,
    downsample_factor: int,
    max_frames: int,
    share: bool,
    file_name: str,
    world_track: bool
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading frames!")
    loader = MegaSamLoader(data_path, track_path, depth_path)
    print(loader.dimension)
    num_frames = min(max_frames, loader.num_frames)

    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider("Point size", min=0.0001, max=0.003, step=0.0001, initial_value=0.001)
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=True)
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=0.1, initial_value=10)
        gui_framerate_options = server.gui.add_button_group("FPS options", ("10", "20", "30", "60"))
        gui_show_occluded = server.gui.add_checkbox("Show Occluded Tracks", True)
        gui_trail_length = server.gui.add_slider("Track Trail Length", min=1, max=100, step=1, initial_value=20)

    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()

    server.scene.set_up_direction('-y')
    server.scene.add_frame("/frames", show_axes=False)

    frame_nodes: list[viser.FrameHandle] = []
    point_nodes: list[viser.PointCloudHandle] = []
    trajectory_nodes: list[list[viser.CurveHandle]] = []

    # save the 3d-track
    tracks_xyz = np.zeros((num_frames, loader.tracks.shape[1], 3), dtype=np.float32)

    extra_pc_data = np.load("/home/jianing/research/3d-motion/3d-tracks/r-tapvid3d_14250544550818363063_880_000_900_000_2_VDacMbTOddDXFtRVMF_m3w.npz")
    extra_points = extra_pc_data["tracks_XYZ"]   # shape: (T, N, 3)
    extra_visibility = extra_pc_data["visibility"]

    extra_points = extra_points[::-1]
    extra_visibility = extra_visibility[::-1]

    cmap = colormaps['turbo']

    for i in tqdm(range(num_frames)):
        position, color = loader.get_points(i, downsample_factor)
        track_position = loader.get_tracks(i, world_track)

        tracks_xyz[i] = track_position

        track_colors = np.tile(np.array([[1.0,0.0,0.0]]), (track_position.shape[0], 1))

        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # point_nodes.append(
        #     server.scene.add_point_cloud(
        #         name=f"/frames/t{i}/point_cloud",
        #         points=position,
        #         colors=color,
        #         point_size=gui_point_size.value,
        #         point_shape="rounded",
        #     )
        # )

        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/tracks",
                points=track_position,
                colors=track_colors,
                point_size=0.001,
                point_shape="rounded",
            )
        )

        # Add frustum
        fov = 2 * np.arctan2(loader.height / 2, loader.K[0, 0])
        aspect = loader.width / loader.height
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            image=loader.rgbd[i, ::downsample_factor, ::downsample_factor, :3],
            wxyz=tf.SO3.from_matrix(loader.cam_c2w[i, :3, :3]).wxyz,
            position=loader.cam_c2w[i, :3, 3],
        )

        server.scene.add_frame(f"/frames/t{i}/frustum/axes", axes_length=0.05, axes_radius=0.005)

        # Inside your per-frame loop:
        traj_handles = []
        if i > 0:
            trail_len = int(gui_trail_length.value)
            sample_stride = 10  # Draw 1 out of every 10 tracks

            for j in range(0, track_position.shape[0], sample_stride):
                if loader.visibility[i, j] or gui_show_occluded.value:
                    traj_segment = tracks_xyz[max(0, i - trail_len): i + 1, j]  # (T, 3)
                    if traj_segment.shape[0] >= 2:
                        # Convert to segments: (N-1, 2, 3)
                        line_segments = np.stack([traj_segment[:-1], traj_segment[1:]], axis=1)
                        # color = np.array(cmap(j / loader.tracks.shape[1])[:3])

                        colors = np.broadcast_to(np.array([[1.0, 0.2, 0.0]]), line_segments.shape)

                        traj = server.scene.add_line_segments(
                            name=f"/frames/t{i}/traj_{j}",
                            points=line_segments,
                            colors=colors,
                            line_width=2.0,
                        )
                        traj_handles.append(traj)

        trajectory_nodes.append(traj_handles)

         # === Add extra track point cloud for this frame ===
        if i < extra_points.shape[0]:
            extra_visible = extra_visibility[i].astype(bool)
            extra_coords = extra_points[i]  # shape: (N, 3)

            # Filter visible points
            visible_coords = extra_coords[extra_visible]
            
            soft_blue = np.array([0.4, 0.7, 1.0])
            extra_colors = np.broadcast_to(soft_blue, visible_coords.shape)

            # Add point cloud of visible tracks
            server.scene.add_point_cloud(
                name=f"/frames/t{i}/extra_tracks",
                points=visible_coords,
                colors=extra_colors,
                point_size=0.001,
                point_shape="rounded",
            )

            # === Add extra trajectory lines (for visible points only) ===
            if i > 0:
                trail_len = int(gui_trail_length.value)
                sample_stride = 10

                for j in range(0, extra_coords.shape[0], sample_stride):
                    if extra_visibility[i, j] or gui_show_occluded.value:
                        traj_segment = extra_points[max(0, i - trail_len): i + 1, j]  # (T, 3)
                        if traj_segment.shape[0] >= 2:
                            # Make segments and color them
                            line_segments = np.stack([traj_segment[:-1], traj_segment[1:]], axis=1)
                            colors = np.broadcast_to(np.array([[0.0, 0.0, 1.0]]), line_segments.shape)

                            server.scene.add_line_segments(
                                name=f"/frames/t{i}/extra_traj_{j}",
                                points=line_segments,
                                colors=colors,
                                line_width=2.0,
                            )

    # Save 3D tracks
    output_path = f"../3d-tracks/{'w-' if world_track else ''}{file_name}.npz"
    np.savez(output_path, tracks_XYZ=tracks_xyz, visibility=loader.visibility.astype(bool))
    print(f"Saved 3D tracks to: {output_path}")

    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update dynamic point size
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        point_nodes[(gui_timestep.value + 1) % num_frames].point_size = gui_point_size.value

        time.sleep(1.0 / gui_framerate.value)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D RGB-D and track data.")

    parser.add_argument(
        "--file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Factor to downsample point cloud resolution",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Maximum number of frames to load",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable sharing via Viser's share URL",
    )
    parser.add_argument(
        "--w",
        action="store_true",
        help="If present, save 3D tracks in world coordinates instead of camera coordinates.",
    )

    args = parser.parse_args()

    data_path = Path(f"/home/jianing/research/3d-motion/mega-sam/outputs_cvd/{args.file}_sgd_cvd_hr.npz")
    track_path = Path(f"/home/jianing/research/3d-motion/co-tracker/2d-track/{args.file}.npz")
    depth_path = Path(f"/home/jianing/research/3d-motion/Video-Depth-Anything/output/{args.file}/{args.file}_depths.npz")

    main(
        data_path=data_path,
        track_path=track_path,
        depth_path=depth_path,
        downsample_factor=args.downsample_factor,
        max_frames=args.max_frames,
        share=args.share,
        file_name=args.file,
        world_track=args.w,
    )
